namespace Lib.Tokenization;

public interface ITokenizer
{
    string GetContractFingerprint();
    int VocabSize { get; }
    int[] Encode(string text);
    string Decode(ReadOnlySpan<int> tokens);
    object GetPayloadForCheckpoint();
}
