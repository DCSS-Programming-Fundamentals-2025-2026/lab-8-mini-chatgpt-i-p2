namespace Lib.MathCore.Sampling;

public class ProbabilitySampler
{
    public int Sample(ReadOnlySpan<float> probs, Random rng)
    {
        double target = rng.NextDouble();

        double cumulativeSum = 0.0;

        for (int i = 0; i < probs.Length; i++)
        {
            cumulativeSum += probs[i];

            if (cumulativeSum > target)
            {
                return i;
            }
        }

        return probs.Length - 1;
    }
}