namespace Lib.MathCore.Calculators;

public class SoftmaxCalculator
{
    public float[] Calculate(ReadOnlySpan<float> logits)
    {
        if (logits.Length == 0)
        {
            return Array.Empty<float>();
        }

        float max = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > max)
            {
                max = logits[i];
            }
        }

        float[] exp = new float[logits.Length];
        float sum = 0;

        for (int i = 0; i < logits.Length; i++)
        {
            exp[i] = MathF.Exp(logits[i] - max);
            
            sum += exp[i];
        }

        for (int i = 0; i < exp.Length; i++)
        {
            exp[i] /= sum;
        }

        return exp;
    }
}