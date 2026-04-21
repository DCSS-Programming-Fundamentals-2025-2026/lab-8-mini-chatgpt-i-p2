using Lib.MathCore;
using Lib.Models.TinyNN.Factories;
using Lib.Models.TinyTransformer.Factories;
using Lib.Runtime;
using Lib.Tokenization;
using MiniChatGPT.ChatConsole;
using MiniChatGPT.Contracts;
using NGram.ModelFactory;
using System.Text;
using System.Text.Json;
using ContractTokenizer = MiniChatGPT.Contracts.ITokenizer;

namespace MiniChatGPT.Chat
{
    class Program
    {
        static int Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            Console.InputEncoding = Encoding.UTF8;

            Console.WriteLine("\nMiniChatGPT Chat");

            string fileName = "checkpoint.json";

            string baseDir = AppContext.BaseDirectory;
            string rootDir = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", ".."));

            string checkpointPath = Path.Combine(rootDir, "data", fileName);

            if (!File.Exists(checkpointPath))
            {
                rootDir = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "..", ".."));
                checkpointPath = Path.Combine(rootDir, "data", fileName);
            }

            if (!File.Exists(checkpointPath))
            {
                Console.WriteLine($"Файл {fileName} не знайдено!");
                Console.WriteLine($"Шлях пошуку: {checkpointPath}");

                return 1;
            }

            try
            {
                JsonCheckpointIO io = new JsonCheckpointIO();
                Checkpoint payload = io.Load(checkpointPath);

                JsonElement tokJson = (JsonElement)payload.TokenizerPayload;
                object rawTokenizer = payload.TokenizerKind == "word"
                    ? new WordTokenizerFactory().FromPayload(tokJson)
                    : new CharTokenizerFactory().FromPayload(tokJson);

                ContractTokenizer tokenizer = new TokenizerAdapter(rawTokenizer);

                JsonElement modJson = (JsonElement)payload.ModelPayload;
                ILanguageModel model = null;
                IMathOps mathOps = new MathOpsImpl();

                if (payload.ModelKind == "trigram")
                {
                    var raw = new NGramModelFactory().CreateTrigramModelFromPayload(tokenizer.VocabSize, modJson);
                    model = new LanguageModelAdapter(raw, "trigram", tokenizer.VocabSize);
                }
                else if (payload.ModelKind == "tinynn")
                {
                    var raw = new TinyNNModelFactory().FromPayload(modJson, tokenizer.VocabSize, mathOps);
                    model = new LanguageModelAdapter(raw, "tinynn", tokenizer.VocabSize);
                }
                else if (payload.ModelKind == "tinytransformer")
                {
                    var raw = TinyTransformerModelFactory.FromPayload(modJson);
                    model = new LanguageModelAdapter(raw, "tinytransformer", tokenizer.VocabSize);
                }
                else if (payload.ModelKind == "bigram")
                {
                    var raw = new NGramModelFactory().CreateBigramModelFromPayload(tokenizer.VocabSize, modJson);
                    model = new LanguageModelAdapter(raw, "bigram", tokenizer.VocabSize);
                }

                if (model == null)
                {
                    throw new Exception($"Модель '{payload.ModelKind}' не підтримується.");
                }

                var modelAdapter = (LanguageModelAdapter)model;

                float temp = modelAdapter.RequiresSoftmax ? 0.7f : 0.3f;
                int topK = 5;

                var sampler = new MiniChatGPT.Sampling.Sampler(mathOps);
                RuntimeTextGenerator generator = new RuntimeTextGenerator(
                    model,
                    sampler,
                    mathOps,
                    tokenizer,
                    modelAdapter.RequiresSoftmax
                );

                Console.WriteLine($"\n [OK] Завантажено: {payload.ModelKind}");
                Console.WriteLine($" [Path] {checkpointPath}");
                Console.WriteLine(" MiniChatGPT активований (введіть текст)");

                new ChatRepl(generator).Run(temp, topK);
            }
            catch (Exception ex)
            {
                Console.WriteLine("\n Помилка ініціалізації:");
                Console.WriteLine(ex.Message);

                return 1;
            }

            return 0;
        }
    }
}