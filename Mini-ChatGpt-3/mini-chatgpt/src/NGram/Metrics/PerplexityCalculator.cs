using System;

namespace NGram.Metrics
{
    public class PerplexityCalculator
    {
        private readonly TrigramModel _model;

        public PerplexityCalculator(TrigramModel trigramModel)
        {
            _model = trigramModel;
        }

        public double ComputePerplexity(ReadOnlySpan<int> tokens)
        {
            if (tokens.Length < 2)
            {
                return 0;
            }

            double logSum = 0;

            for (int i = 1; i < tokens.Length; i++)
            {
                ReadOnlySpan<int> context = tokens.Slice(0, i);
                float[] scores = _model.NextTokenScores(context);
                float prob = scores[tokens[i]];

                if (prob == 0)
                {
                    prob = 1e-10f;
                }

                logSum += Math.Log(prob);
            }

            int words = tokens.Length - 1;
            double avgLog = logSum / words;
            double perplexity = Math.Exp(-avgLog);

            return perplexity;
        }

        public static double ComputePerplexity(ReadOnlySpan<int> tokens, Func<int[], float[]> nextScore)
        {
            if (tokens.Length < 2)
            {
                return 0;
            }

            double logSum = 0;

            for (int i = 1; i < tokens.Length; i++)
            {
                int[] context = tokens.Slice(0, i).ToArray();
                float[] scores = nextScore(context);
                float prob = scores[tokens[i]];

                if (prob == 0)
                {
                    prob = 1e-10f;
                }

                logSum += Math.Log(prob);
            }

            int n = tokens.Length - 1;

            return Math.Exp(-logSum / n);
        }
    }
}
