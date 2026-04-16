using Lib.MathCore;
using Lib.Models.TinyNN;
using Lib.Models.TinyNN.Configuration;
using Lib.MathCore;
using Lib.Models.TinyNN;
using Lib.Models.TinyNN.Configuration;
using Lib.Tokenization;
using MiniChatGPT.Contracts;
using System.Text.Json;

namespace Lib.Training
{
    public class TinyNNTrainer
    {
        public void Run(int[] trainTokens, TinyNNModel model, WordTokenizer tokenizer, string dataFolderPath, int epochs, float learningRate, int contextSize)
        {
            Console.WriteLine("\nТренуємо TinyNN");
            Console.WriteLine($"Навчання на {epochs} епох | LR: {learningRate}");

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                float totalLoss = 0;
                int steps = 0;

                for (int i = 0; i < trainTokens.Length - contextSize; i++)
                {
                    int[] context = trainTokens[i..(i + contextSize)];
                    int target = trainTokens[i + contextSize];

                    float loss = model.TrainStep(context, target, learningRate);

                    totalLoss += loss;
                    steps++;
                }

                if (epoch % 5 == 0 || epoch == 1 || epoch == epochs)
                {
                    Console.WriteLine($"Епоха {epoch}/{epochs} | Середня помилка (Loss): {totalLoss / steps:F6}");
                }
            }

            Checkpoint cp = new Checkpoint(
                "tinynn",
                "neural_v1_trained",
                model.ToPayload(),
                tokenizer.GetPayloadForCheckpoint(),
                42,
                "word"
            );

            SaveCheckpoint(cp, dataFolderPath);
        }

        private void SaveCheckpoint(Checkpoint cp, string dataFolderPath)
        {
            var options = new JsonSerializerOptions { WriteIndented = false };
            string jsonString = JsonSerializer.Serialize(cp, options);

            string path = Path.Combine(dataFolderPath, "checkpoint_nn.json");

            File.WriteAllText(path, jsonString);
            Console.WriteLine($"\nTinyNN збережена: {path}");
        }
    }
}
