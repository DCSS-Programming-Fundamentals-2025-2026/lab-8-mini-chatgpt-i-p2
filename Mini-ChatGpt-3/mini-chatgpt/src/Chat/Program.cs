using System.Text;
using System.Text.Json;
using ContractTokenizer = MiniChatGPT.Contracts.ITokenizer;
using MiniChatGPT.Contracts;
using Lib.MathCore; 
using Lib.Tokenization;
using Lib.Runtime;
using NGram.ModelFactory;
using MiniChatGPT.ChatConsole;

namespace MiniChatGPT.Chat
{
    class Program
    {
        static int Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            Console.InputEncoding = Encoding.UTF8;

            string fileName = "checkpoint.json";
            string checkpointPath = "";
            float temp = 0.3f;
            int topK = 5;

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
                object rawTokenizer = payload.TokenizerKind == "word"? new WordTokenizerFactory().FromPayload(tokJson) : new CharTokenizerFactory().FromPayload(tokJson);

                ContractTokenizer tokenizer = new TokenizerAdapter(rawTokenizer);

                JsonElement modJson = (JsonElement)payload.ModelPayload;
                ILanguageModel model = payload.ModelKind == "bigram"? new NGramModelFactory().CreateBigramModelFromPayload(tokenizer.VocabSize, modJson): new NGramModelFactory().CreateTrigramModelFromPayload(tokenizer.VocabSize, modJson);

                IMathOps mathOps = new MiniChatGPT.MathCore.MathOps();
                MiniChatGPT.Sampling.Interfaces.ISampler sampler = new MiniChatGPT.Sampling.Sampler(mathOps);

                RuntimeTextGenerator generator = new RuntimeTextGenerator(
                    model,
                    sampler,
                    mathOps,
                    tokenizer,
                    false
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