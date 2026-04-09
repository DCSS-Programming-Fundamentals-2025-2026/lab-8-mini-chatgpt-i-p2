using System.Text;
using System.Text.Json;
using ContractTokenizer = MiniChatGPT.Contracts.ITokenizer;
using MiniChatGPT.Contracts;
using Lib.MathCore;
using Lib.Tokenization;
using Lib.Runtime;
using NGram.ModelFactory;
using MiniChatGPT.ChatConsole;
using Lib.Models.TinyNN.Factories;
using Lib.Models.TinyTransformer.Factories;

namespace MiniChatGPT.Chat
{
    class Program
    {
        static int Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            Console.InputEncoding = Encoding.UTF8;

            string checkpointPath = "";
            float temp = 0.3f;
            int topK = 5;

            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--checkpoint")
                {
                    if (i + 1 < args.Length)
                    {
                        checkpointPath = args[i + 1];
                    }
                }
            }

            if (string.IsNullOrEmpty(checkpointPath))
            {
                string fileName = "checkpoint.json";
                string[] possiblePaths =
                {
                    Path.Combine(Environment.CurrentDirectory, "data", fileName),
                    Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "data", fileName)),
                    Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "data", fileName))
                };

                foreach (string path in possiblePaths)
                {
                    if (File.Exists(path))
                    {
                        checkpointPath = path;
                        break;
                    }
                }
            }

            if (string.IsNullOrEmpty(checkpointPath))
            {
                Console.WriteLine("Не знайдено файл конфігурації.");
                return 1;
            }

            try
            {
                JsonCheckpointIO io = new JsonCheckpointIO();
                Checkpoint payload = io.Load(checkpointPath);

                JsonElement tokJson = (JsonElement)payload.TokenizerPayload;
                object rawTokenizer;

                if (payload.TokenizerKind == "word")
                {
                    rawTokenizer = new WordTokenizerFactory().FromPayload(tokJson);
                }
                else
                {
                    rawTokenizer = new CharTokenizerFactory().FromPayload(tokJson);
                }

                ContractTokenizer tokenizer = new TokenizerAdapter(rawTokenizer);

                JsonElement modJson = (JsonElement)payload.ModelPayload;
                ILanguageModel model = null;
                bool requiresSoftmax = false;
                IMathOps mathOps = new MathOpsImpl();

                if (payload.ModelKind == "bigram")
                {
                    model = new NGramModelFactory().CreateBigramModelFromPayload(tokenizer.VocabSize, modJson);
                }
                else if (payload.ModelKind == "trigram")
                {
                    model = new NGramModelFactory().CreateTrigramModelFromPayload(tokenizer.VocabSize, modJson);
                }
                else if (payload.ModelKind == "tinynn")
                {
                    object rawTinyNN = new TinyNNModelFactory().FromPayload(modJson, tokenizer.VocabSize, mathOps);
                    model = new LanguageModelAdapter(rawTinyNN, "tinynn", tokenizer.VocabSize);
                    requiresSoftmax = true;
                }
                else if (payload.ModelKind == "tinytransformer")
                {
                    object rawTransformer = TinyTransformerModelFactory.FromPayload(modJson);
                    model = new LanguageModelAdapter(rawTransformer, "tinytransformer", tokenizer.VocabSize);
                    requiresSoftmax = true;
                }
                else
                {
                    Console.WriteLine("Невідомий тип моделі.");
                    return 1;
                }

                MiniChatGPT.Sampling.Interfaces.ISampler sampler = new MiniChatGPT.Sampling.Sampler(mathOps);

                RuntimeTextGenerator generator = new RuntimeTextGenerator(
                    model,
                    sampler,
                    mathOps,
                    tokenizer,
                    requiresSoftmax
                );

                Console.WriteLine("\nСистема MiniChatGPT активована");
                new ChatRepl(generator).Run(temp, topK, payload.Seed);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nПомилка: {ex.Message}");
                return 1;
            }

            return 0;
        }
    }
}