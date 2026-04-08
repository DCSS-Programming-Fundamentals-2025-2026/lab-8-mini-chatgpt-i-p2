using Lib.Models.TinyTransformer.Configuration;

namespace Lib.Models.TinyTransformer.Layers;

public class TransformerBlock
{
    private readonly TinyTransformerConfig _config;
    public float[] Forward(int[] context)
    {
        SelfAttentionLayer selfAttentionLayer = new SelfAttentionLayer(_config);
        FeedForwardLayer feedForwardLayer = new FeedForwardLayer(_config);
        
        return feedForwardLayer.Project(selfAttentionLayer.Compute(context));
    }

    public TransformerBlock(TinyTransformerConfig config)
    {
        _config = config;
    }
}