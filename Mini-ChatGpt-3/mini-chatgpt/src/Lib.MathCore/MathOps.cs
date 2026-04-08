using Lib.MathCore;

namespace MiniChatGPT.MathCore
{
    public class MathOps : IMathOps
    {
        public float[] Softmax(ReadOnlySpan<float> scores)
        {
            if (scores.IsEmpty)
            {
                return Array.Empty<float>();
            }

            float maxScore = scores[0];
            for (int i = 1; i < scores.Length; i++)
            {
                if (scores[i] > maxScore)
                {
                    maxScore = scores[i];
                }
            }

            float[] exps = new float[scores.Length];
            float sum = 0f;

            for (int i = 0; i < scores.Length; i++)
            {
                exps[i] = (float)Math.Exp(scores[i] - maxScore);
                sum += exps[i];
            }

            for (int i = 0; i < scores.Length; i++)
            {
                exps[i] /= sum;
            }

            return exps;
        }

        public int ArgMax(ReadOnlySpan<float> probabilities)
        {
            if (probabilities.IsEmpty)
            {
                return -1;
            }

            int maxIndex = 0;
            float maxVal = probabilities[0];

            for (int i = 1; i < probabilities.Length; i++)
            {
                if (probabilities[i] > maxVal)
                {
                    maxVal = probabilities[i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        }

        public float CrossEntropyLoss(ReadOnlySpan<float> probabilities, int targetIndex)
        {
            if (targetIndex < 0 || targetIndex >= probabilities.Length)
            {
                return 100f;
            }

            float prob = probabilities[targetIndex];

            if (prob < 1e-10f)
            {
                prob = 1e-10f;
            }

            return -(float)Math.Log(prob);
        }

        public int SampleFromProbs(ReadOnlySpan<float> probabilities, Random rng)
        {
            float r = (float)rng.NextDouble();
            float cumulative = 0f;

            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulative += probabilities[i];
                if (r <= cumulative)
                {
                    return i;
                }
            }

            return probabilities.Length - 1;
        }

        public string GetContractFingerprint()
        {
            return "MathOps-v1.2-ReadOnlySpan";
        }
    }
}