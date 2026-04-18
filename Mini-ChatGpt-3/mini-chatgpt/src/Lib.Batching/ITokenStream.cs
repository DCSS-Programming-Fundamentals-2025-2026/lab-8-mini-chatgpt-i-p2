using System;

namespace Lib.Batching
{
    public interface ITokenStream
    {
        int Length { get; }
        ReadOnlySpan<int> GetTokens();
    }
}