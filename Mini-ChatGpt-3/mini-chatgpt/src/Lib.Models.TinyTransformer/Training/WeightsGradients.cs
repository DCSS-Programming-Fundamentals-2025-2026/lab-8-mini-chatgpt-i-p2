namespace Lib.Models.TinyTransformer.Training;

public class WeightsGradients
{
    public float[][] dOutputW;
    public float[][] dFfn2;
    public float[][] dFfn1;
    public float[] dFfn1Bias;
    public float[] dFfn2Bias;
    public float[] dOutputBias;

    public float[][] dQ;
    public float[][] dV;
    public float[][] dK;
    public float[][] dO;

    public WeightsGradients(int embeddingSize, int vocabSize)
    {
        dOutputW = new float[embeddingSize][];
        dFfn1 = new float[embeddingSize][];
        dQ = new float[embeddingSize][];
        dV = new float[embeddingSize][];
        dK = new float[embeddingSize][];
        dO = new float[embeddingSize][];
        for (int i = 0; i < embeddingSize; i++)
        {
            dOutputW[i] = new float[vocabSize];
            dFfn1[i] = new float[embeddingSize];
            dQ[i] = new float[embeddingSize];
            dV[i] = new float[embeddingSize];
            dK[i] = new float[embeddingSize];
            dO[i] = new float[embeddingSize];
        }

        dFfn2 = new float[embeddingSize * 4][];
        for (int i = 0; i < embeddingSize * 4; i++)
        {
            dFfn2[i] = new float[embeddingSize];
        }

        dFfn1Bias = new float[embeddingSize * 4];
        dFfn2Bias = new float[embeddingSize];
        dOutputBias = new float[vocabSize];
    }
}