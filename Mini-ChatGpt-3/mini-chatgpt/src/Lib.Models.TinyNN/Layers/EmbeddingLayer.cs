using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;

namespace Lib.Models.TinyNN.Layers;

public static class EmbeddingLayer
{
    public static float[] EncodeContext(ReadOnlySpan<int> context, TinyNNWeights weights, TinyNNConfig config)
    {
        var activeContext = context;
        
        if (context.Length > config.ContextSize)
        {
            activeContext = context[^config.ContextSize..];
        }
        
        float[] hidden = new float[config.EmbeddingSize];

        if (activeContext.Length == 0)
        {
            return hidden;
        }
        
        foreach (int wordId in activeContext)
        {
            float[] wordVector = weights.Embeddings[wordId];
            
            for (int i = 0; i < config.EmbeddingSize; i++)
            {
                hidden[i] += wordVector[i];
            }
        }

        for (int i = 0; i < hidden.Length; i++)
        {
            hidden[i] /= activeContext.Length;
        }
        
        return hidden;
    }

    public static void Backward(float[] dHidden, ReadOnlySpan<int> context, TinyNNWeights weights, TinyNNConfig config, float lr)
    {
        var activeContext = context;
        
        if (context.Length > config.ContextSize)
        {
            activeContext = context[^config.ContextSize..];
        }

        if (activeContext.Length == 0)
        {
            return;
        }
        
        float step = lr / activeContext.Length;

        foreach (int wordId in activeContext)
        {
            for (int i = 0; i < config.EmbeddingSize; i++)
            {
                weights.Embeddings[wordId][i] -= dHidden[i] * step;
            }
        }
    }
}