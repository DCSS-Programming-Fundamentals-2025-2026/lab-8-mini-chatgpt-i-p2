namespace Lib.Models.TinyTransformer.Training;

public class TrainingCache
{
    public int[] Context { get; set; }
    public float[][] X { get; set; }
    public float[][] Q { get; set; }
    public float[][] K { get; set; }
    public float[][] V { get; set; }
    public float[][] Attn { get; set; }
    public float[][] Proj { get; set; }
    public float[][] OutMatrix { get; set; }
    public float[] Hidden { get; set; }
    public float[] FirstLinearOutput { get; set; }
    public float[] ReluOutput { get; set; }
    public float[] SecondLinearOutput { get; set; }
}