using System.Text.Json.Serialization;
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.DummyInterfaces;
using Lib.Models.TinyNN.Layers;
using Lib.Models.TinyNN.State;
using Lib.MathCore;

namespace Lib.Models.TinyNN;

public class TinyNNModel  : ILanguageModel
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

            if (i == target)
            {
                dLogits[i] -= 1f;
            }

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
}

public record TinyNNPayload(
    [property: JsonPropertyName("config")] TinyNNConfig Config, 
    [property: JsonPropertyName("weights")] TinyNNWeights Weights
);