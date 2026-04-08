using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Enums;
using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Layers;

public class FeedForwardLayer
{
    private readonly TinyTransformerConfig _config;

    public FeedForwardLayer(TinyTransformerConfig config)
    {
        _config = config;
    }

    public float[] Project(float[] hidden)
    {
        float[][] hiddenHelper = new float[1][];
        hiddenHelper[0] = hidden;

        float[][] firstLinear = Linear(hiddenHelper, LinearAction.Expanse);
        Relu(firstLinear[0]);
        float[][] secondLinear = Linear(firstLinear, LinearAction.Compress);

        float[][] thirdLinear = Linear(secondLinear, LinearAction.Vocab);

        return thirdLinear[0];
    }

    public float[][] Linear(float[][] matrix, LinearAction linear)
    {
        switch (linear)
        {
            case LinearAction.Expanse:
                var firstLinear = MatrixHelper.MultiplyMatrix(matrix, _config.Weights.ffn1);
                MatrixHelper.LineSumm(firstLinear[0], _config.Weights.ffn1Bias);
                return firstLinear;

            case LinearAction.Compress:
                var secondLinear = MatrixHelper.MultiplyMatrix(matrix, _config.Weights.ffn2);
                MatrixHelper.LineSumm(secondLinear[0], _config.Weights.ffn2Bias);
                return secondLinear;

            case LinearAction.Vocab:
                var thirdLinear = MatrixHelper.MultiplyMatrix(matrix, _config.Weights.OutputW);
                MatrixHelper.LineSumm(thirdLinear[0], _config.Weights.OutputBias);
                return thirdLinear;

            default: throw new NotImplementedException();
        }
    }

    public static void Relu(float[] array)
    {
        for (int i = 0; i < array.Length; i++)
        {
            if (array[i] < 0)
            {
                array[i] = 0;
            }
        }
    }
}