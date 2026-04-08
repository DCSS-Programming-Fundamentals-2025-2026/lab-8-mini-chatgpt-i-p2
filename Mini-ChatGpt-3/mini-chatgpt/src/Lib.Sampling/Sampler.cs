using MiniChatGPT.Sampling.Interfaces;
using MiniChatGPT.Sampling.Processing;
using Lib.MathCore;

namespace MiniChatGPT.Sampling
{
    public class Sampler : ISampler
    {
        private readonly IMathOps _mathOps;

        public Sampler(IMathOps mathOps)
        {
            if (mathOps == null)
            {
                throw new ArgumentNullException("mathOps");
            }

            _mathOps = mathOps;
        }

        public int Sample(float[] probs, float temperature, int topK, Random? rng = null)
        {
            if (probs == null || probs.Length == 0)
            {
                throw new ArgumentException("Масив ймовірностей є порожнім!", nameof(probs));
            }

            if (rng == null)
            {
                rng = new Random();
            }

            if (temperature < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(temperature), "Температура не може бути від'ємною!");
            }

            if (temperature == 0f)
            {
                return _mathOps.ArgMax(probs);
            }

            if (topK <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(topK), "Значення Top-K має бути більшим за нуль!");
            }

            float[] tempered = TemperatureScaler.Scale(probs, temperature);

            int[] idx = TopKSelector.Select(tempered, topK);

            float[] topKProbs = new float[idx.Length];

            for (int i = 0; i < idx.Length; i++)
            {
                topKProbs[i] = MathF.Exp(tempered[idx[i]]); 
            }

            ProbabilityNormalizer.Normalize(topKProbs);

            int selectedKIdx = _mathOps.SampleFromProbs(topKProbs, rng);

            return idx[selectedKIdx];
        }

        public int SampleWithSeed(float[] probs, float temperature, int topK, int? seed)
        {
            Random rng;

            if (seed.HasValue)
            {
                rng = new Random(seed.Value);
            }
            else
            {
                rng = null; 
            }

            return Sample(probs, temperature, topK, rng);
        }
    }
}
