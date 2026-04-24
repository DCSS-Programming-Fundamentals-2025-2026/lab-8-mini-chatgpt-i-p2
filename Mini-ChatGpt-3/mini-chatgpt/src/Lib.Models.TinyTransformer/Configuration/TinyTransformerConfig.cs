using Lib.MathCore;
using Lib.Models.TinyTransformer.State;
using System.Text.Json.Serialization;

namespace Lib.Models.TinyTransformer.Configuration;

public class TinyTransformerConfig
{
    public int VocabSize { get; set; }
    public int EmbeddingSize { get; set; }
    public int HeadCount { get; set; }
    public int ContextSize { get; set; }
    public IMathOps MathOps { get; set; }
    public int Seed { get; set; }
    public TinyTransformerWeights Weights { get; set; }
    public float[][] TokenEmbeddings { get; set; }

    [JsonConstructor]
    public TinyTransformerConfig()
    {
    }
    
    public TinyTransformerConfig(int vocabSize, int embeddingSize = 16, int headCount = 1, int contextSize = 8,
        int seed = 42, TinyTransformerWeights weights = null, float[][] tokenEmbeddings = null)

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
        else Weights = weights;

        if (tokenEmbeddings == null)
        {
            TokenEmbeddings = TinyTransformerWeights.GenerateMatrix(vocabSize, embeddingSize);
        }
        else TokenEmbeddings = tokenEmbeddings;
    }
}