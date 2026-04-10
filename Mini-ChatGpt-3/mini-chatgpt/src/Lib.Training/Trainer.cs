using Lib.Tokenization;
using MiniChatGPT.Contracts;
using NGram;
using NGram.ModelFactory;
using System.Text;

namespace Lib.Training
{
    class Trainer
    {
        static void Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            Console.InputEncoding = Encoding.UTF8;

            Console.WriteLine("Починаємо тренування моделі...");

            CorpusClass corpus = CorpusLoader.Load("data/showcase.txt");

            WordTokenizer tokenizer = WordTokenizer.BuildFromText(corpus.TrainText);

            int[] trainTokens = tokenizer.Encode(corpus.TrainText);

            Console.WriteLine($"Словник створено. Розмір: {tokenizer.VocabSize}");

            NGramModelFactory factory = new NGramModelFactory();
            TrigramModel tModel = factory.CreateTrigramModel(tokenizer.VocabSize);

            Console.WriteLine("Модель тріграм була створена");

            Console.WriteLine("Тренування поч");

            tModel.Train(trainTokens);

            Console.WriteLine("Тренування кінець");

            Checkpoint cp = new Checkpoint(
                tModel.ModelKind,
                tModel.GetContractFingerprint(),
                tModel.GetPayloadForCheckpoint(),
                tokenizer.GetPayloadForCheckpoint(),
                42,
                "word"
            );

            System.Text.Json.JsonSerializerOptions options = new System.Text.Json.JsonSerializerOptions()
            {
                WriteIndented = true
            };
            string jsonString = System.Text.Json.JsonSerializer.Serialize(cp, options);

            jsonString = System.Text.RegularExpressions.Regex.Replace(
                jsonString,
                @"\[(?:\s*[0-9\.\-]+,\s*)*\s*[0-9\.\-]+\s*\]",
                m => m.Value.Replace("\r", "").Replace("\n", "").Replace("  ", " ")
            );

            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string rootDir = Path.GetFullPath(Path.Combine(baseDir, @"..\..\..\..\..\"));
            string dataFolderPath = Path.Combine(rootDir, "data");

            if (!System.IO.Directory.Exists(dataFolderPath))
            {
                System.IO.Directory.CreateDirectory(dataFolderPath);
            }

            string path = Path.Combine(dataFolderPath, "checkpoint.json");

            System.IO.File.WriteAllText(path, jsonString);

            Console.WriteLine($"Ідеальна матриця збережена в корінь ---> ({path})");
        }
    }
}
