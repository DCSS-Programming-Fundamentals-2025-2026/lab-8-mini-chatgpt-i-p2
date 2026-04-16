using Lib.MathCore;
using Lib.Models.TinyNN;
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;
using Lib.Tokenization;
using MiniChatGPT.Contracts;
using System.Text;

namespace Lib.Training
{
    public class Trainer
    {
        static void Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            Console.InputEncoding = Encoding.UTF8;

            Console.WriteLine("\nТренування ГПТ");

            string baseDir = AppContext.BaseDirectory;
            string rootDir = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", ".."));
            string dataFolderPath = Path.Combine(rootDir, "data");

            if (!Directory.Exists(dataFolderPath))
            {
                Directory.CreateDirectory(dataFolderPath);
            }

            CorpusClass corpus = CorpusLoader.Load(Path.Combine(dataFolderPath, "showcase.txt"));
            WordTokenizer tokenizer = WordTokenizer.BuildFromText(corpus.TrainText);
            int[] trainTokens = tokenizer.Encode(corpus.TrainText);

            bool isRunning = true;

            while (isRunning)
            {
                Console.WriteLine("\nТип навчання:");
                Console.WriteLine("1. Trigram");
                Console.WriteLine("2. TinyNN");
                Console.WriteLine("3. Завершити тренування");
                Console.Write("\nВибір: ");

                string choice = Console.ReadLine();

                switch (choice)
                {
                    case "1":
                        new TrigramTrainer().Run(trainTokens, tokenizer, dataFolderPath);
                        break;
                    case "2":
                        RunTinyNNWorkflow(tokenizer, trainTokens, dataFolderPath);
                        break;
                    case "3":
                        Console.WriteLine("Тренування завершено");
                        isRunning = false;
                        break;
                    default:
                        Console.WriteLine("Немає такої опції");
                        break;
                }
            }
        }

        static void RunTinyNNWorkflow(WordTokenizer tokenizer, int[] tokens, string dataFolderPath)
        {
            string checkpointPath = Path.Combine(dataFolderPath, "checkpoint_nn.json");
            TinyNNModel model;
            var mathOps = new MathOpsImpl();
            int contextSize = 5;
            int embeddingSize = 64;

            if (File.Exists(checkpointPath))
            {
                try
                {
                    Console.WriteLine("Завантаження ваг для навчання...");
                    var io = new JsonCheckpointIO();
                    var payload = io.Load(checkpointPath);

                    if (payload.ModelKind == "tinynn")
                    {
                        var factory = new Lib.Models.TinyNN.Factories.TinyNNModelFactory();
                        model = (TinyNNModel)factory.FromPayload((System.Text.Json.JsonElement)payload.ModelPayload, tokenizer.VocabSize, mathOps);
                    }
                    else
                    {
                        model = CreateNewModel(tokenizer.VocabSize, embeddingSize, contextSize, mathOps);
                    }
                }
                catch
                {
                    Console.WriteLine("Помилка завантаження. Створено нову модель");
                    model = CreateNewModel(tokenizer.VocabSize, embeddingSize, contextSize, mathOps);
                }
            }
            else
            {
                model = CreateNewModel(tokenizer.VocabSize, embeddingSize, contextSize, mathOps);
            }

            Console.Write("Скільки епох вчимо? (стандартно 50): ");
            if (!int.TryParse(Console.ReadLine(), out int epochs))
            {
                epochs = 50;
            }

            var trainer = new TinyNNTrainer();
            trainer.Run(
                tokens,
                model,
                tokenizer,
                dataFolderPath,
                epochs: epochs,
                learningRate: 0.005f, 
                contextSize: contextSize
            );
        }

        static TinyNNModel CreateNewModel(int vocab, int embed, int context, IMathOps math)
        {
            var config = new TinyNNConfig(vocab, embed, context);

            return new TinyNNModel(config, new TinyNNWeights(config), math);
        }
    }
}