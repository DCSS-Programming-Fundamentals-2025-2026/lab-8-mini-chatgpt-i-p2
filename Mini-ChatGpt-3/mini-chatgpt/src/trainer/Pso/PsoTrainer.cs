using MiniChatGpt.Contracts;

namespace Trainer.Pso;

using System;
public class PsoTrainer
{
    public void Train(IPsoOptimizable baseModel, List<int> dataTokens, int swarmSize = 30, int epochs = 300)
        {
            float[] initialWeights = baseModel.GetFlatWeights();
            int paramCount = initialWeights.Length;
            Console.WriteLine($"[PSO] Start. Parameters: {paramCount}. Swarm: {swarmSize} particles.");

            var rnd = new Random();
            var swarm = new List<Particle>();
            for (int i = 0; i < swarmSize; i++)
            {
                swarm.Add(new Particle(paramCount, rnd));
            }

            float[] globalBestPosition = new float[paramCount];
            float globalBestLoss = float.MaxValue;
            object lockObj = new object();

            float w = 0.729f;  
            float c1 = 1.494f;
            float c2 = 1.494f; 

            int batchSize = Math.Min(500, dataTokens.Count);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                int startIndex = rnd.Next(0, dataTokens.Count - batchSize);
                var batchTokens = dataTokens.GetRange(startIndex, batchSize);

                Parallel.ForEach(swarm, particle =>
                {
                    var localModel = baseModel.Clone(); 
                    localModel.SetFlatWeights(particle.Position);
                    
                    float currentLoss = localModel.CalculateTotalLoss(batchTokens);

                    if (currentLoss < particle.BestLoss)
                    {
                        particle.BestLoss = currentLoss;
                        Array.Copy(particle.Position, particle.BestPosition, paramCount);
                    }

                    lock (lockObj)
                    {
                        if (currentLoss < globalBestLoss)
                        {
                            globalBestLoss = currentLoss;
                            Array.Copy(particle.Position, globalBestPosition, paramCount);
                        }
                    }
                });

                foreach (var p in swarm)
                {
                    for (int i = 0; i < paramCount; i++)
                    {
                        float r1 = (float)rnd.NextDouble();
                        float r2 = (float)rnd.NextDouble();

                        p.Velocity[i] = w * p.Velocity[i] + 
                                        c1 * r1 * (p.BestPosition[i] - p.Position[i]) + 
                                        c2 * r2 * (globalBestPosition[i] - p.Position[i]);

                        p.Position[i] += p.Velocity[i];
                    }
                }

                Console.WriteLine($"Epoch {epoch + 1}/{epochs} | Best Loss: {globalBestLoss}");
            }

            baseModel.SetFlatWeights(globalBestPosition);
            Console.WriteLine("Training complete.");
        }
}