using Lib.Tokenization;
using MiniChatGPT.Contracts;
using NGram;
using NGram.ModelFactory;
using System.Text.Json;

namespace Lib.Training
{
    public class TrigramTrainer
    {
        public void Run(int[] trainTokens, WordTokenizer tokenizer, string dataFolderPath)
        {
            NGramModelFactory factory = new NGramModelFactory();
            TrigramModel tModel = factory.CreateTrigramModel(tokenizer.VocabSize);

            Console.WriteLine("\nТренуємо тріграм");
            tModel.Train(trainTokens);

            Checkpoint cp = new Checkpoint(
                tModel.ModelKind,
                tModel.GetContractFingerprint(),
                tModel.GetPayloadForCheckpoint(),
                tokenizer.GetPayloadForCheckpoint(),
                42,
                "word"
            );

            SaveCheckpoint(cp, dataFolderPath);
        }

        private void SaveCheckpoint(Checkpoint cp, string dataFolderPath)
        {
            var options = new JsonSerializerOptions { WriteIndented = true };
            string jsonString = JsonSerializer.Serialize(cp, options);

            string path = Path.Combine(dataFolderPath, "checkpoint_trigram.json");

            jsonString = System.Text.RegularExpressions.Regex.Replace(
                jsonString,
                @"\[(?:\s*[0-9\.\-]+,\s*)*\s*[0-9\.\-]+\s*\]",
                m => m.Value.Replace("\r", "").Replace("\n", "").Replace("  ", " ")
            );

            File.WriteAllText(path, jsonString);
            Console.WriteLine($"Trigram збережена: {path}");
        }
    }
}