<<<<<<< HEAD
﻿using System.Data;
=======
﻿using Lib.MathCore;
>>>>>>> origin/TinyTransformerTrainingAndIntegreation
using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Training;

public class Training
{
<<<<<<< HEAD
    public static void Train(TinyTransformerModel model, int[] context, int targetIndex, float learningRate)
    {
        TrainingCache cache = new TrainingCache();
        var logits = model.NextTokenScores(context, true, cache);
        
        var loss = CalculateLoss(logits, targetIndex);

        WeightsGradients weightsGradients = new WeightsGradients(model.config.EmbeddingSize, model.config.VocabSize);
        
        model.BackPropagation(loss, cache, weightsGradients);
        
        Update(model.config.Weights, weightsGradients, learningRate);
=======
    public static void Train(TinyTransformerModel model, ReadOnlySpan<int> context, int targetIndex, float learningRate)
    {
        TrainingCache cache = new TrainingCache();
        var logits = model.NextTokenScores(context, true, cache);
        var probabilites = new MathOpsImpl().Softmax(logits);
        var gradient = CalculateLoss(probabilites, targetIndex);

        WeightsGradients weightsGradients = new WeightsGradients(model._config.EmbeddingSize, model._config.VocabSize);

        model.BackPropagation(gradient, cache, weightsGradients);

        Update(model._config.TokenEmbeddings, model._config.Weights, weightsGradients, learningRate);
>>>>>>> origin/TinyTransformerTrainingAndIntegreation
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
<<<<<<< HEAD
    
    public static void Update(TinyTransformerWeights weights, WeightsGradients grads, float learningRate)
    {
=======

    public static void Update(float[][] tE, TinyTransformerWeights weights, WeightsGradients grads, float learningRate)
    {
        UpdateMatrix(tE, grads.dE, learningRate);
>>>>>>> origin/TinyTransformerTrainingAndIntegreation
        UpdateMatrix(weights.wQ, grads.dQ, learningRate);
        UpdateMatrix(weights.wK, grads.dK, learningRate);
        UpdateMatrix(weights.wV, grads.dV, learningRate);
        UpdateMatrix(weights.wO, grads.dO, learningRate);

        UpdateMatrix(weights.ffn1, grads.dFfn1, learningRate);
        UpdateMatrix(weights.ffn2, grads.dFfn2, learningRate);
        UpdateMatrix(weights.OutputW, grads.dOutputW, learningRate);

        UpdateVector(weights.ffn1Bias, grads.dFfn1Bias, learningRate);
        UpdateVector(weights.ffn2Bias, grads.dFfn2Bias, learningRate);
        UpdateVector(weights.OutputBias, grads.dOutputBias, learningRate);
    }

    private static void UpdateMatrix(float[][] matrix, float[][] gradient, float learningRate)
    {
        for (int i = 0; i < matrix.Length; i++)
        {
            for (int j = 0; j < matrix[i].Length; j++)
            {
                matrix[i][j] -= learningRate * gradient[i][j];
            }
        }
    }

    private static void UpdateVector(float[] vector, float[] gradient, float learningRate)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] -= learningRate * gradient[i];
        }
    }
}