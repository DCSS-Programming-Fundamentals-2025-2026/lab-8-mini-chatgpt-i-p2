using Lib.MathCore;
using Lib.Models.TinyNN.Factories;
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
            Console.WriteLine("Оберіть модель для завантаження:");
            Console.WriteLine("1. Trigram");
            Console.WriteLine("2. TinyNN");
            Console.Write("\nВибір: ");

            string choice = Console.ReadLine();
            string fileName;

            if (choice == "2")
            {
                fileName = "checkpoint_nn.json";
            }
            else
            {
                fileName = "checkpoint_trigram.json";
            }

            string baseDir = AppContext.BaseDirectory;
            string rootDir = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", ".."));
            string checkpointPath = Path.Combine(rootDir, "data", fileName);

            float temp;
            if (choice == "2")
            {
                temp = 0.7f;
            }
            else
            {
                temp = 0.3f;
            }
            int topK = 5;

            if (!File.Exists(checkpointPath))
            {
                Console.WriteLine($"Файл {fileName} не знайдено!");
                Console.WriteLine($"Шлях: {checkpointPath}");
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

                if (payload.ModelKind == "trigram")
                {
                    model = new NGramModelFactory().CreateTrigramModelFromPayload(tokenizer.VocabSize, modJson);
                    requiresSoftmax = false;
                }
                else if (payload.ModelKind == "tinynn")
                {
                    object rawTinyNN = new TinyNNModelFactory().FromPayload(modJson, tokenizer.VocabSize, mathOps);
                    model = new LanguageModelAdapter(rawTinyNN, "tinynn", tokenizer.VocabSize);
                    requiresSoftmax = true;
                }

                var sampler = new MiniChatGPT.Sampling.Sampler(mathOps);
                RuntimeTextGenerator generator = new RuntimeTextGenerator(
                    model,
                    sampler,
                    mathOps,
                    tokenizer,
                    requiresSoftmax
                );

                Console.WriteLine($"\n Завантажена модель: {fileName}");
                Console.WriteLine("MiniChatGPT активований");

                new ChatRepl(generator).Run(temp, topK, payload.Seed);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Помилка завантаження:");
                Console.WriteLine(ex.Message);
                return 1;
            }

            return 0;
        }
    }
}