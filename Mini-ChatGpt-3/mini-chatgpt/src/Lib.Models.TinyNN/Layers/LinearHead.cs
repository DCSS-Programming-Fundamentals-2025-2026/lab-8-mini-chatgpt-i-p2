using Lib.Models.TinyNN.Configuration; 
using Lib.Models.TinyNN.State;

namespace Lib.Models.TinyNN.Layers;

public static class LinearHead
{
    public static float[] Project(float[] hidden, TinyNNWeights weights, TinyNNConfig config)
    {
        float[] logits = new float[config.VocabSize];

        for (int i = 0; i < config.VocabSize; i++)
        {
            float sum = 0f;
            for (int j = 0; j < config.EmbeddingSize; j++)
            {
                sum += hidden[j] * weights.OutputWeights[j][i];
            }
            sum+=weights.OutputBias[i];
            logits[i] = sum;
        }
        
        return logits;
    }

    public static float[] Backward(float[] dLogits, float[] hidden, TinyNNWeights weights, TinyNNConfig config, float lr)
    {
        float[] dHidden = new float[config.EmbeddingSize];

        for (int i = 0; i < config.VocabSize; i++)
        {
            float step = dLogits[i] * lr;
            weights.OutputBias[i] -= step;

            for (int j = 0; j < config.EmbeddingSize; j++)
            {
                dHidden[j] += dLogits[i] * weights.OutputWeights[j][i];
                weights.OutputWeights[j][i] -= hidden[j] * step;
            }
        }
        
        return dHidden;
    }
}