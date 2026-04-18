namespace Lib.Batching.Streams
{
    public class ArrayTokenStream : ITokenStream
    {
        private readonly int[] _tokens;

        public ArrayTokenStream(int[] tokens)
        {
            _tokens = tokens ?? throw new ArgumentNullException(nameof(tokens));
        }

        public int Length => _tokens.Length;

        public ReadOnlySpan<int> GetTokens() => _tokens.AsSpan();
    }
}

