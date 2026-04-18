using System;

namespace Lib.Batching.Sampling
{
    public static class BatchWindowSampler
    {
        public static int GetRandomStartIndex(int totalTokens, int blockSize, Random rng)
        {
            if (totalTokens <= 1)
            {
                throw new InvalidOperationException("Недостатньо токенів для формування батчу.");
            }
            if (totalTokens <= blockSize)
            {
                return 0;
            }
            int maxStartIndex = totalTokens - blockSize - 1;
            return rng.Next(0, maxStartIndex + 1);
        }
    }
}