
using Lib.MathCore.Calculators;
using Lib.MathCore.Sampling;
using Lib.MathCore.Utilities;

namespace Lib.MathCore;

public class MathOpsImpl : IMathOps
{
    
    private readonly SoftmaxCalculator _softmax = new();
    private readonly LossCalculator _loss = new();
    private readonly ProbabilitySampler _sampler = new();

    
    public float[] Softmax(ReadOnlySpan<float> logits)
    {
        return _softmax.Calculate(logits);
    }

    public float CrossEntropyLoss(ReadOnlySpan<float> logits, int target)
    {
        return _loss.Calculate(logits, target);
    }

    public int ArgMax(ReadOnlySpan<float> scores)
    {
        return ScoreUtilities.GetArgMax(scores);
    }

    public int SampleFromProbs(ReadOnlySpan<float> probs, Random rng)
    {
        return _sampler.Sample(probs, rng);
    }
}