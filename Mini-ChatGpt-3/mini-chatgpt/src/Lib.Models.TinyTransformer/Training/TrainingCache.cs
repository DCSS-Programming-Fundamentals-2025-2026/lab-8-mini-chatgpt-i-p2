namespace Lib.Models.TinyTransformer.Training;

public class TrainingCache
{
    public float[][] X;
    public float[][] Q;
    public float[][] K;
    public float[][] V;
    public float[][] Attn;
    public float[][] Proj;
    public float[] Hidden;
    public float[] FirstLinearOutput;
    public float[] ReluOutput;
    public float[] SecondLinearOutput;
    public float[] ThirdLinearOutput;
}