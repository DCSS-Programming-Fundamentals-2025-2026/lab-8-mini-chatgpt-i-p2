namespace MiniChatGPT.Contracts;

/// <summary>Interface for language models. Implemented by Lib.Models.NGram, TinyNN, TinyTransformer.</summary>
public interface ILanguageModel : IContractFingerprint
{
    string ModelKind { get; }
    int VocabSize { get; }
    float[] NextTokenScores(ReadOnlySpan<int> context);
    object GetPayloadForCheckpoint();
}