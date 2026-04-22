using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Enums;
using Lib.Models.TinyTransformer.State;
using Lib.Models.TinyTransformer.Training;

namespace Lib.Models.TinyTransformer.Layers;

public class FeedForwardLayer
{
    private readonly TinyTransformerConfig _config;

    public FeedForwardLayer(TinyTransformerConfig config)
    {
        _config = config;
    }

    public float[] Project(float[] hidden, bool isTraining = false, TrainingCache cache = null)
    {
        float[][] hiddenHelper = new float[1][];
        hiddenHelper[0] = hidden;

        float[][] firstLinear = Linear(hiddenHelper, LinearAction.Expanse);
        if(isTraining) cache.FirstLinearOutput = firstLinear[0];
        
        Relu(firstLinear[0]);
        
        float[][] secondLinear = Linear(firstLinear, LinearAction.Compress);

        float[][] thirdLinear = Linear(secondLinear, LinearAction.Vocab);
        if(isTraining)
        {
            cache.ReluOutput = firstLinear[0];
            cache.SecondLinearOutput = secondLinear[0];
            cache.ThirdLinearOutput = thirdLinear[0];
        }
        
        return thirdLinear[0];
    }

    public float[] Backward(float[] gradient, TrainingCache cache, WeightsGradients weightsGradients)
    {
        var secondGrad = LinearBackward(cache.ThirdLinearOutput, gradient,
            _config.Weights.OutputW, 
            weightsGradients.dOutputW, weightsGradients.dOutputBias);
        
        var thirdGrad = LinearBackward(cache.SecondLinearOutput, secondGrad,
            _config.Weights.ffn2, 
            weightsGradients.dFfn2, weightsGradients.dFfn2Bias);
        
        for (int i = 0; i < thirdGrad.Length; i++)
        {
            if (cache.ReluOutput[i] == 0.0f) 
            {
                thirdGrad[i] = 0.0f;
            }
        }
        
        var fourthGrad = LinearBackward(cache.FirstLinearOutput, thirdGrad,
            _config.Weights.ffn1, 
            weightsGradients.dFfn1, weightsGradients.dFfn1Bias);
        
        return fourthGrad;
    }
    
    public static float[] LinearBackward(
        float[] inputX,
        float[] gradOutput,
        float[][] weights,
        float[][] weightGrads,
        float[] biasGrads
    )
    {
        int inSize = inputX.Length;
        int outSize = gradOutput.Length;

        for (int i = 0; i < inSize; i++)
        {
            for (int j = 0; j < outSize; j++)
            {
                weightGrads[i][j] += inputX[i] * gradOutput[j];
            }
        }

        for (int j = 0; j < outSize; j++)
        {
            biasGrads[j] += gradOutput[j];
        }

        float[] gradInput = new float[inSize];
        for (int i = 0; i < inSize; i++)
        {
            for (int j = 0; j < outSize; j++)
            {
                gradInput[i] += gradOutput[j] * weights[i][j];
            }
        }
        
        return gradInput;
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