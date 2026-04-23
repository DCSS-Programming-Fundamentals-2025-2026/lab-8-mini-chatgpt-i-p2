using Lib.MathCore;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Enums;
using Lib.Models.TinyTransformer.State;
using Lib.Models.TinyTransformer.Training;


namespace Lib.Models.TinyTransformer.Layers;

public class SelfAttentionLayer
{
    private readonly TinyTransformerConfig _config;
    private readonly MathOpsImpl MathOps = new();

    public SelfAttentionLayer(TinyTransformerConfig config)
    {
        _config = config;
    }

    public float[] Compute(int[] context, bool isTraining = false, TrainingCache cache = null)
    {
        //step 1
        float[][] x = InitXmatrix(context);

        //step 2
        float[][] Q = InitMatrix(x, QKV.Q);
        float[][] K = InitMatrix(x, QKV.K);
        float[][] V = InitMatrix(x, QKV.V);

        //step 3
        float[][] scores = MatrixHelper.MultiplyMatrix(Q, MatrixHelper.TransposeMatrix(K));
        EachElementDivideBySquareRootOfEmbeddingSizeWithMask(scores);

        //step 4
        float[][] attn = SoftmaxEachRow(scores);

        //step 5
        float[][] outMatrix = WeightedSum(attn, V);

        //step 6
        float[][] proj = MatrixHelper.MultiplyMatrix(outMatrix, _config.Weights.wO);

        if (isTraining)
        {
            cache.X = x.Select(row => row.ToArray()).ToArray();
            cache.Q = Q.Select(row => row.ToArray()).ToArray();
            cache.K = K.Select(row => row.ToArray()).ToArray();
            cache.V = V.Select(row => row.ToArray()).ToArray();
            cache.Attn = attn.Select(row => row.ToArray()).ToArray();
            cache.OutMatrix = outMatrix.Select(row => row.ToArray()).ToArray();
            cache.Proj = proj.Select(row => row.ToArray()).ToArray();
        }

        //step 7
        return proj[proj.Length - 1];
    }

    public void Backward(float[] fourthGrad, TrainingCache cache, WeightsGradients weightsGrads)
    {
        float[] gradOutMatrixRow = AttentionBackward(
            cache.OutMatrix[cache.OutMatrix.Length - 1],
            fourthGrad,
            _config.Weights.wO, weightsGrads.dO
        );
        
        int contextLen = cache.Attn.Length;
        float[] gradAttnRow = new float[contextLen];
        float[][] gradV = new float[contextLen][];

        for (int j = 0; j < contextLen; j++)
        {
            gradV[j] = new float[_config.EmbeddingSize];
            float weight = cache.Attn[contextLen - 1][j];
    
            for (int k = 0; k < _config.EmbeddingSize; k++)
            {
                gradV[j][k] = gradOutMatrixRow[k] * weight;
        
                gradAttnRow[j] += gradOutMatrixRow[k] * cache.V[j][k];
            }
        }
        
        float sumGradAttn = 0;
        for (int j = 0; j < contextLen; j++) 
            sumGradAttn += cache.Attn[contextLen - 1][j] * gradAttnRow[j];

        float[] gradScoresRow = new float[contextLen];
        for (int j = 0; j < contextLen; j++)
        {
            float p = cache.Attn[contextLen - 1][j];
            gradScoresRow[j] = p * (gradAttnRow[j] - sumGradAttn);
    
            gradScoresRow[j] /= MathF.Sqrt(_config.EmbeddingSize);
        }

        float[][] gradQRow = MatrixHelper.MultiplyMatrix(new float[][]{gradScoresRow}, cache.K);
        float[][] gradK = new float[contextLen][];
        for (int j = 0; j < contextLen; j++)
        {
            gradK[j] = new float[_config.EmbeddingSize];
            for (int k = 0; k < _config.EmbeddingSize; k++)
            {
                gradK[j][k] = gradScoresRow[j] * cache.Q[contextLen - 1][k];
            }
        }
        AttentionBackward(cache.X[contextLen - 1], gradQRow[0], _config.Weights.wQ, weightsGrads.dQ);

        for(int i = 0; i < contextLen; i++)
            AttentionBackward(cache.X[i], gradK[i], _config.Weights.wK, weightsGrads.dK);

        for(int i = 0; i < contextLen; i++)
            AttentionBackward(cache.X[i], gradV[i], _config.Weights.wV, weightsGrads.dV);
    }

    public static float[] AttentionBackward(
        float[] inputX,
        float[] gradOutput,
        float[][] weights,
        float[][] weightGrads
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


    public float[][] InitXmatrix(int[] context)
    {
        float[][] x = new float[context.Length > _config.ContextSize ? _config.ContextSize : context.Length][];

        if (context.Length > _config.ContextSize)
        {
            context = context.TakeLast(_config.ContextSize).ToArray();
        }

        for (int i = 0; i < x.Length; i++)
        {
            x[i] = _config.TokenEmbeddings[context[i]];
        }

        return x;
    }

    public float[][] InitMatrix(float[][] x, QKV qkv)
    {
        switch (qkv)
        {
            case QKV.K:
                return MatrixHelper.MultiplyMatrix(x, _config.Weights.wK);
            case QKV.Q:
                return MatrixHelper.MultiplyMatrix(x, _config.Weights.wQ);
            case QKV.V:
                return MatrixHelper.MultiplyMatrix(x, _config.Weights.wV);
            default:
                throw new ArgumentOutOfRangeException(nameof(qkv), qkv, null);
        }
    }

    public void EachElementDivideBySquareRootOfEmbeddingSizeWithMask(float[][] matrix)
    {
        for (int i = 0; i < matrix.Length; i++)
        {
            for (int j = 0; j < matrix[i].Length; j++)
            {
                if (j <= i)
                {
                    matrix[i][j] /= MathF.Sqrt(_config.EmbeddingSize);
                    continue;
                }

                matrix[i][j] = float.NegativeInfinity;
            }
        }
    }

    public float[][] SoftmaxEachRow(float[][] matrix)
    {
        float[][] res = new float[matrix.Length][];

        for (int i = 0; i < matrix.Length; i++)
        {
            res[i] = MathOps.Softmax(matrix[i]);
        }

        return res;
    }

    public float[][] WeightedSum(float[][] attn, float[][] V)
    {
        float[][] res = new float[attn.Length][];

        for (int i = 0; i < attn.Length; i++)
        {
            float[] sumVector = new float[_config.EmbeddingSize];

            for (int j = 0; j < attn[i].Length; j++)
            {
                float weight = attn[i][j];
                float[] valueVect = V[j];

                for (int k = 0; k < _config.EmbeddingSize; k++)
                {
                    sumVector[k] += weight * valueVect[k];
                }
            }

            res[i] = sumVector;
        }

        return res;
    }
}