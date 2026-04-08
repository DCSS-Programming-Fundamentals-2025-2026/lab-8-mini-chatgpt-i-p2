namespace Lib.MathCore.Calculators;

public class LossCalculator
{
    private readonly SoftmaxCalculator _softmax = new();

    public float Calculate(ReadOnlySpan<float> logits, int target)
    {
        float[] probabilities = _softmax.Calculate(logits);

        float p = probabilities[target];


        return -MathF.Log(p + 1e-10f);
    }
}