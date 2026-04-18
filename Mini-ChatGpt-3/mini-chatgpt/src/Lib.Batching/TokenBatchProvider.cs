using Lib.Batching.Sampling;

namespace Lib.Batching
{
    public class TokenBatchProvider : IBatchProvider
    {
        private readonly ITokenStream _stream;

        public ITokenStream Stream => _stream;

        public TokenBatchProvider(ITokenStream stream)
        {
            _stream = stream ?? throw new ArgumentNullException(nameof(stream));
        }

        public Batch GetBatch(int batchSize, int blockSize, Random? rng)
        {
            var random = rng ?? new Random();
            var tokens = _stream.GetTokens();
            var totalTokens = tokens.Length;

            if (totalTokens <= 1)
            {
                throw new InvalidOperationException("Потік токенів занадто короткий.");
            }

            int[][] contexts = new int[batchSize][];
            int[] targets = new int[batchSize];

            int actualBlockSize = Math.Min(blockSize, totalTokens - 1);

            for (int i = 0; i < batchSize; i++)
            {
                int startIndex = BatchWindowSampler.GetRandomStartIndex(totalTokens, actualBlockSize, random);

                contexts[i] = new int[blockSize];

                int offset = blockSize - actualBlockSize;

                for (int j = 0; j < actualBlockSize; j++)
                {
                    contexts[i][offset + j] = tokens[startIndex + j];
                }

                targets[i] = tokens[startIndex + actualBlockSize];
            }

            return new Batch(contexts, targets);
        }
    }
}