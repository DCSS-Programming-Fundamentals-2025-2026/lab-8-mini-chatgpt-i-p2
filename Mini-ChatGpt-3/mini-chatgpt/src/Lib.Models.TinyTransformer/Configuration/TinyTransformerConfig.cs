using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Configuration;

public class TinyTransformerConfig
{
    public float[][] TokenEmbeddings = [];
    public int VocabSize { get; set; }
    public int EmbeddingSize { get; set; }
    public int HeadCount { get; set; }
    public int ContextSize { get; set; }
    public int Seed { get; set; }
    public TinyTransformerWeights Weights { get; set; }

    public TinyTransformerConfig(int vocabSize, int embeddingSize = 16, int headCount = 1, int contextSize = 8,
        int seed = 42, TinyTransformerWeights weights = null)
    {
        VocabSize = vocabSize;
        EmbeddingSize = embeddingSize;
        HeadCount = headCount;
        ContextSize = contextSize;
        Seed = seed;
        if (weights == null)
        {
            Weights = new TinyTransformerWeights(embeddingSize, vocabSize);
        }
        else
        {
            Weights = weights;
        }
    }
}