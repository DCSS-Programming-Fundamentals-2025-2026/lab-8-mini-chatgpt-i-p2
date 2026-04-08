using Lib.MathCore;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Enums;
using Lib.Models.TinyTransformer.State;


namespace Lib.Models.TinyTransformer.Layers;

public class SelfAttentionLayer
{
    private readonly TinyTransformerConfig _config;
    private readonly MathOpsImpl MathOps = new ();
    public SelfAttentionLayer(TinyTransformerConfig config)
    {
        _config = config;
    }
    
    public float[] Compute(int[] context)
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

        //step 7
        return proj[proj.Length - 1];
    }

    public float[][] InitXmatrix(int[] context)
    {
        float[][] x = new float[context.Length > _config.ContextSize ? _config.ContextSize : context.Length][];

        if (context.Length > 8)
        {
            context = context.TakeLast(8).ToArray();
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
