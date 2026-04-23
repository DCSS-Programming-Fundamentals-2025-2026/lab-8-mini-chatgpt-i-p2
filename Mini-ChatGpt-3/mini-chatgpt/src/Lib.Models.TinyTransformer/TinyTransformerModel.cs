using Lib.MathCore;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Layers;
using Lib.Models.TinyTransformer.Training;

namespace Lib.Models.TinyTransformer;

public class TinyTransformerModel
{
    public TinyTransformerConfig _config;
    private readonly MathOpsImpl _mathOps = new MathOpsImpl();
    private readonly string Version = "1.0.0";
    public float[] NextTokenScores(ReadOnlySpan<int> context, bool isTraining = false, TrainingCache cache = null)
    {
        TransformerBlock block = new TransformerBlock(_config);    
            
        return _mathOps.Softmax(block.Forward(context.ToArray(), isTraining, cache));
    }

    public void BackPropagation(float[] gradient, TrainingCache cache, WeightsGradients weightsGradients)
    {
        TransformerBlock block = new TransformerBlock(_config);
        block.Backward(gradient, cache, weightsGradients);
    }

    public TinyTransformerModel(TinyTransformerConfig config)
    {
        this._config = config;
    }
    
    public TinyTransformerConfig GetPayloadForCheckpoint()
    { 
        return _config;   
    }
    
    public string GetContractFingerprint()
    {
        return this.Version;
    }
}