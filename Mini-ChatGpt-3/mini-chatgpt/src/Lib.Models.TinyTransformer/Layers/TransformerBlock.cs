using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Training;

namespace Lib.Models.TinyTransformer.Layers;

public class TransformerBlock
{
    private readonly TinyTransformerConfig _config;
    public float[] Forward(int[] context, bool isTraining = false, TrainingCache cache = null)
    {
        SelfAttentionLayer selfAttentionLayer = new SelfAttentionLayer(_config);
        FeedForwardLayer feedForwardLayer = new FeedForwardLayer(_config);
        
        return feedForwardLayer.Project(selfAttentionLayer.Compute(context, isTraining, cache), isTraining, cache);
    }

    public void Backward(float[] gradient, TrainingCache cache, WeightsGradients weightsGradients)
    {
        SelfAttentionLayer selfAttentionLayer = new SelfAttentionLayer(_config);
        FeedForwardLayer feedForwardLayer = new FeedForwardLayer(_config);
        
        var ffnGradient = feedForwardLayer.Backward(gradient, cache, weightsGradients);
        selfAttentionLayer.Backward(ffnGradient, cache, weightsGradients);
    }

    public TransformerBlock(TinyTransformerConfig config)
    {
        _config = config;
    }
}