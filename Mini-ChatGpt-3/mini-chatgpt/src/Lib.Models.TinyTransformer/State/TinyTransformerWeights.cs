using Lib.Models.TinyTransformer.Configuration;

namespace Lib.Models.TinyTransformer.State;

public class TinyTransformerWeights
{
    public static Random rnd = new Random(42);
    public float[][] wQ { get; set; }
    public float[][] wK { get; set; }
    public float[][] wV { get; set; }
    public float[][] wO { get; set; }
    public float[][] OutputW { get; set; }
    public float[][] ffn1 { get; set; }
    public float[][] ffn2 { get; set; }
    public float[] ffn1Bias { get; set; }
    public float[] ffn2Bias { get; set; }
    public float[] OutputBias { get; set; }

    public static float[][] GenerateMatrix(int rows, int cols)
    {
        float[][] res = new float[rows][];

        for (int i = 0; i < rows; i++)
        {
            res[i] = new float[cols];

            for (int j = 0; j < cols; j++)
            {
                res[i][j] = rnd.NextSingle() * 0.2f - 0.1f;
            }
        }

        return res;
    }

    public TinyTransformerWeights(int embeddingSize, int vocabSize)
    {
        wQ = GenerateMatrix(embeddingSize, embeddingSize);

        wK = GenerateMatrix(embeddingSize, embeddingSize);
        wV = GenerateMatrix(embeddingSize, embeddingSize);
        wO = GenerateMatrix(embeddingSize, embeddingSize);
        OutputW = GenerateMatrix(embeddingSize, vocabSize);

        ffn1 =
            GenerateMatrix(embeddingSize, embeddingSize * 4);

        ffn2 =
            GenerateMatrix(embeddingSize * 4, embeddingSize);

        ffn1Bias = new float[embeddingSize * 4];
        ffn2Bias = new float[embeddingSize];
        OutputBias = new float[vocabSize];
    }
}