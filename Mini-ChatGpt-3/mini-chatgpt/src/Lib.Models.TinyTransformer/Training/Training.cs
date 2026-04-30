using Lib.MathCore;
using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Training;

public class Training
{
    public static float ForwardBackward(TinyTransformerModel model, ReadOnlySpan<int> context, int targetIndex, WeightsGradients weightsGradients)
    {
        TrainingCache cache = new TrainingCache();
        var logits = model.NextTokenScores(context, true, cache);
        float loss = new MathOpsImpl().CrossEntropyLoss(logits, targetIndex);

        var probabilities = new MathOpsImpl().Softmax(logits);
        var gradient = CalculateLoss(probabilities, targetIndex);

        model.BackPropagation(gradient, cache, weightsGradients);

        return loss;
    }

    public static float[] CalculateLoss(float[] logits, int targetIndex)
    {
        float[] loss = new float[logits.Length];

        for (int i = 0; i < logits.Length; i++)
        {
            loss[i] = (i == targetIndex) ? logits[i] - 1.0f : logits[i];
        }

        return loss;
    }

    public static void Update(float[][] tE, TinyTransformerWeights weights, WeightsGradients grads, float learningRate, int batchSize)
    {
        float scale = learningRate / batchSize;

        ClipGradients(grads, 1.0f, batchSize);

        UpdateMatrix(tE, grads.dE, scale);
        UpdateMatrix(weights.wQ, grads.dQ, scale);
        UpdateMatrix(weights.wK, grads.dK, scale);
        UpdateMatrix(weights.wV, grads.dV, scale);
        UpdateMatrix(weights.wO, grads.dO, scale);

        UpdateMatrix(weights.ffn1, grads.dFfn1, scale);
        UpdateMatrix(weights.ffn2, grads.dFfn2, scale);
        UpdateMatrix(weights.OutputW, grads.dOutputW, scale);

        UpdateVector(weights.ffn1Bias, grads.dFfn1Bias, scale);
        UpdateVector(weights.ffn2Bias, grads.dFfn2Bias, scale);
        UpdateVector(weights.OutputBias, grads.dOutputBias, scale);
    }

    private static void ClipGradients(WeightsGradients grads, float maxNorm, int batchSize)
    {
        float normSq = 0f;

        normSq += MatrixNormSq(grads.dE);
        normSq += MatrixNormSq(grads.dQ);
        normSq += MatrixNormSq(grads.dK);
        normSq += MatrixNormSq(grads.dV);
        normSq += MatrixNormSq(grads.dO);
        normSq += MatrixNormSq(grads.dFfn1);
        normSq += MatrixNormSq(grads.dFfn2);
        normSq += MatrixNormSq(grads.dOutputW);
        normSq += VectorNormSq(grads.dFfn1Bias);
        normSq += VectorNormSq(grads.dFfn2Bias);
        normSq += VectorNormSq(grads.dOutputBias);

        float norm = MathF.Sqrt(normSq) / batchSize;

        if (norm > maxNorm)
        {
            float clipScale = maxNorm / norm;
            ScaleMatrix(grads.dE, clipScale);
            ScaleMatrix(grads.dQ, clipScale);
            ScaleMatrix(grads.dK, clipScale);
            ScaleMatrix(grads.dV, clipScale);
            ScaleMatrix(grads.dO, clipScale);
            ScaleMatrix(grads.dFfn1, clipScale);
            ScaleMatrix(grads.dFfn2, clipScale);
            ScaleMatrix(grads.dOutputW, clipScale);
            ScaleVector(grads.dFfn1Bias, clipScale);
            ScaleVector(grads.dFfn2Bias, clipScale);
            ScaleVector(grads.dOutputBias, clipScale);
        }
    }

    private static float MatrixNormSq(float[][] m)
    {
        float sum = 0f;
        for (int i = 0; i < m.Length; i++)
            for (int j = 0; j < m[i].Length; j++)
                sum += m[i][j] * m[i][j];
        return sum;
    }

    private static float VectorNormSq(float[] v)
    {
        float sum = 0f;
        for (int i = 0; i < v.Length; i++)
            sum += v[i] * v[i];
        return sum;
    }

    private static void ScaleMatrix(float[][] m, float s)
    {
        for (int i = 0; i < m.Length; i++)
            for (int j = 0; j < m[i].Length; j++)
                m[i][j] *= s;
    }

    private static void ScaleVector(float[] v, float s)
    {
        for (int i = 0; i < v.Length; i++)
            v[i] *= s;
    }

    private static void UpdateMatrix(float[][] matrix, float[][] gradient, float scale)
    {
        for (int i = 0; i < matrix.Length; i++)
        {
            for (int j = 0; j < matrix[i].Length; j++)
            {
                matrix[i][j] -= scale * gradient[i][j];
            }
        }
    }

    private static void UpdateVector(float[] vector, float[] gradient, float scale)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] -= scale * gradient[i];
        }
    }
}