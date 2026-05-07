using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.DummyInterfaces;
using Lib.Models.TinyNN.Layers;
using Lib.Models.TinyNN.State;
using Lib.MathCore;
using MiniChatGpt.Contracts;

namespace Lib.Models.TinyNN;
public class TinyNNModel : ILanguageModel, IPsoOptimizable
{
    private readonly TinyNNConfig _config;
    private readonly TinyNNWeights _weights;
    private readonly IMathOps _mathOps;
    
    public string GetContractFingerprint() => "TinyNN-A7-v1";
    public string ModelKind => "tinynn";
    public int VocabSize => _config.VocabSize;

    public TinyNNModel(TinyNNConfig config, TinyNNWeights weights, IMathOps mathOps)
    {
        _config = config;
        _weights = weights;
        _mathOps = mathOps;
    }


    
    public float[] NextTokenScores(ReadOnlySpan<int> context)
    {
        if (context.Length > _config.ContextSize)
        {
            context = context.Slice(context.Length - _config.ContextSize);
        }
        var hidden = EmbeddingLayer.EncodeContext(context, _weights, _config);
        var logits = LinearHead.Project(hidden, _weights, _config);
        
        return logits;
    }

    public float TrainStep(ReadOnlySpan<int> context, int target, float lr)
    {
        var logits = NextTokenScores(context);
        var probs = _mathOps.Softmax(logits);
        var loss = _mathOps.CrossEntropyLoss(logits, target);
        
        float[] dLogits = new float[_config.VocabSize];

        for (int i = 0; i < _config.VocabSize; i++)
        {
            dLogits[i] = probs[i];
            if (i == target) dLogits[i] -= 1f;
        }
        
        var hidden = EmbeddingLayer.EncodeContext(context, _weights, _config);
        var dHidden = LinearHead.Backward(dLogits, hidden, _weights, _config, lr);
        
        EmbeddingLayer.Backward(dHidden, context, _weights, _config, lr);

        return loss;
    }

    public TinyNNPayload ToPayload()
    {
        return new TinyNNPayload(_config, _weights);
    }

    // PSO 

    public float[] GetFlatWeights()
    {
        int embSize = _config.VocabSize * _config.EmbeddingSize;
        int outSize = _config.EmbeddingSize * _config.VocabSize;
        int biasSize = _config.VocabSize;
    
        float[] flat = new float[embSize + outSize + biasSize];
        int offset = 0;

        for (int i = 0; i < _config.VocabSize; i++)
        {
            for (int j = 0; j < _config.EmbeddingSize; j++)
            {
                flat[offset++] = _weights.Embeddings[i][j];
            }
        }

        for (int i = 0; i < _config.EmbeddingSize; i++)
        {
            for (int j = 0; j < _config.VocabSize; j++)
            {
                flat[offset++] = _weights.OutputWeights[i][j];
            }
        }

        for (int i = 0; i < _config.VocabSize; i++)
        {
            flat[offset++] = _weights.OutputBias[i];
        }

        return flat;
    }

    public void SetFlatWeights(float[] flat)
    {
        int offset = 0;

        for (int i = 0; i < _config.VocabSize; i++)
        {
            for (int j = 0; j < _config.EmbeddingSize; j++)
            {
                _weights.Embeddings[i][j] = flat[offset++];
            }
        }

        for (int i = 0; i < _config.EmbeddingSize; i++)
        {
            for (int j = 0; j < _config.VocabSize; j++)
            {
                _weights.OutputWeights[i][j] = flat[offset++];
            }
        }

        for (int i = 0; i < _config.VocabSize; i++)
        {
            _weights.OutputBias[i] = flat[offset++];
        }
    }
    
    
    public float CalculateTotalLoss(List<int> tokens)
    {
        if (tokens == null || tokens.Count < 2) return 0f;

        float totalLoss = 0f;
        int count = 0;

        for (int i = 0; i < tokens.Count - 1; i++)
        {
            int start = Math.Max(0, i - _config.ContextSize + 1);
            int length = i - start + 1;
            
            int[] contextArr = new int[length];
            tokens.CopyTo(start, contextArr, 0, length);
            
            int target = tokens[i + 1];

            var logits = NextTokenScores(contextArr);
            totalLoss += _mathOps.CrossEntropyLoss(logits, target);
            count++;
        }

        return count > 0 ? totalLoss / count : 0f;
    }

    public IPsoOptimizable Clone()
    {
        var clonedWeights = DeepCopyWeights(this._weights);
        return new TinyNNModel(this._config, clonedWeights, this._mathOps);
    }

    private TinyNNWeights DeepCopyWeights(TinyNNWeights original)
    {
        var copy = new TinyNNWeights(); 

        copy.Embeddings = new float[original.Embeddings.Length][];
        for (int i = 0; i < original.Embeddings.Length; i++)
        {
            copy.Embeddings[i] = new float[original.Embeddings[i].Length];
            Buffer.BlockCopy(original.Embeddings[i], 0, copy.Embeddings[i], 0, original.Embeddings[i].Length * sizeof(float));
        }

        copy.OutputWeights = new float[original.OutputWeights.Length][];
        for (int i = 0; i < original.OutputWeights.Length; i++)
        {
            copy.OutputWeights[i] = new float[original.OutputWeights[i].Length];
            Buffer.BlockCopy(original.OutputWeights[i], 0, copy.OutputWeights[i], 0, original.OutputWeights[i].Length * sizeof(float));
        }

        copy.OutputBias = new float[original.OutputBias.Length];
        Buffer.BlockCopy(original.OutputBias, 0, copy.OutputBias, 0, original.OutputBias.Length * sizeof(float));

        return copy;
    }
}

public record TinyNNPayload(
    [property: JsonPropertyName("config")] TinyNNConfig Config, 
    [property: JsonPropertyName("weights")] TinyNNWeights Weights
);