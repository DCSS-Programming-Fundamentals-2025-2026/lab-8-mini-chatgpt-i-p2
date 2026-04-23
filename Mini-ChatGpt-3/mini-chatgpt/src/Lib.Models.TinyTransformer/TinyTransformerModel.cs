using Lib.MathCore;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Layers;
using Lib.Models.TinyTransformer.Training;

namespace Lib.Models.TinyTransformer;

public class TinyTransformerModel
{
    public TinyTransformerConfig config;
    private readonly MathOpsImpl _mathOps = new ();
    public float[] NextTokenScores(int[] context, bool isTraining = false, TrainingCache cache = null)
    {
        TransformerBlock block = new TransformerBlock(config);    
            
        return _mathOps.Softmax(block.Forward(context, isTraining, cache));
    }

    public void BackPropagation(float[] gradient, TrainingCache cache, WeightsGradients weightsGradients)
    {
        TransformerBlock block = new TransformerBlock(config);
        block.Backward(gradient, cache, weightsGradients);
    }

    public TinyTransformerModel(TinyTransformerConfig config)
    {
        this.config = config;
    }
    
    public TinyTransformerConfig GetPayloadForCheckpoint()
    { 
        return config;   
    }
}