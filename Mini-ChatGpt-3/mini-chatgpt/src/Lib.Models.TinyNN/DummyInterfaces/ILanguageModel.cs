namespace Lib.Models.TinyNN.DummyInterfaces;

public interface ILanguageModel
{
    string ModelKind { get; }
    int VocabSize { get; }
    
    float[] NextTokenScores(ReadOnlySpan<int> context);
    float TrainStep(ReadOnlySpan<int> context, int target, float lr);
    TinyNNPayload ToPayload();
}