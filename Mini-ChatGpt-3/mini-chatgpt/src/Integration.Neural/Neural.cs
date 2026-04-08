using System.Text.Json;
using Lib.MathCore;
using Lib.Models.TinyNN;
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.Factories;
using Lib.Models.TinyNN.Layers;
using Lib.Models.TinyNN.State;
using Lib.Models.TinyTransformer;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Enums;
using Lib.Models.TinyTransformer.Layers;

namespace Integration.Neural;



public class Tests
{
    public TinyTransformerConfig config;
    public SelfAttentionLayer selfAttentionLayer;
    public FeedForwardLayer feedForwardLayer;
    
    [SetUp]
    public void Setup()
    {
        float[][] tokenEmbeddings = new float[][]
        {
            new float[]
            {
                -0.0251f, 0.0901f, 0.0464f, 0.0197f, -0.0688f, -0.0688f, -0.0884f, 0.0732f,
                0.0112f, -0.0441f, 0.0821f, -0.0034f, 0.0556f, -0.0912f, 0.0223f, -0.0771f
            },
            new float[]
            {
                0.0202f, 0.0416f, -0.0959f, 0.0940f, 0.0665f, -0.0575f, -0.0636f, -0.0633f,
                -0.0128f, 0.0883f, -0.0334f, 0.0412f, -0.0091f, 0.0772f, -0.0551f, 0.0119f
            },
            new float[]
            {
                -0.0392f, 0.0050f, -0.0136f, -0.0418f, 0.0224f, -0.0721f, -0.0416f, -0.0267f,
                0.0911f, -0.0223f, 0.0554f, -0.0882f, 0.0129f, -0.0443f, 0.0667f, -0.0031f
            }
        };

        config = new(tokenEmbeddings.Length)
        {
            TokenEmbeddings = tokenEmbeddings
        };
        
        selfAttentionLayer = new SelfAttentionLayer(config);
        feedForwardLayer = new FeedForwardLayer(config);
    }
    
    [Test]
    public void Integration_TrainStep_ExecutesPipelineAndUpdatesWeights()
    {
        var config = new TinyNNConfig (10,4,3);
        var weights = new TinyNNWeights(config);
        var fakeMath = new MathOpsImpl();
        var model = new TinyNNModel(config, weights, fakeMath);
        
        int[] context = new[] {1, 2, 5};
        int target = 7; //<=8
        float lr = 0.1f;
        float initialWeight = weights.OutputWeights[0][0];

        var loss = model.TrainStep(context, target, lr); //=0.99f

        Assert.Multiple(() =>
        {
            Assert.That(loss, Is.EqualTo(0.99f));
            Assert.That(weights.OutputWeights[0][0], Is.Not.EqualTo(initialWeight));
        });
    }
    
    [Test]
    public void Checkpoint_RoundTrip_RestoresModelFunctionality()
    {
        var config = new TinyNNConfig(VocabSize: 5, EmbeddingSize: 4, ContextSize: 3);
        var weights = new TinyNNWeights(config);
        var fakeMath = new MathOpsImpl();
        var originalModel = new TinyNNModel(config, weights, fakeMath);
    
        int[] context = { 1, 2 };
        var originalLogits = originalModel.NextTokenScores(context);

        var payloadObj = originalModel.ToPayload();
    
        string jsonString = JsonSerializer.Serialize(payloadObj);

        using var jsonDocument = JsonDocument.Parse(jsonString);
        var rootElement = jsonDocument.RootElement;
    
        var factory = new TinyNNModelFactory();
        var restoredModel = factory.FromPayload(rootElement, config.VocabSize, fakeMath);

        var restoredLogits = restoredModel.NextTokenScores(context);

        Assert.That(restoredLogits, Is.EqualTo(originalLogits));
    }
    
    [Test]
    public void Project_CalculatesCorrectLogits()
    {
        var config = new TinyNNConfig (2,2,2);
        var weights = new TinyNNWeights(config);

        float[] hidden = { 1.0f, 2.0f };
        weights.OutputBias = new[] { 0.1f, 0.2f };

        weights.OutputWeights[0] = new[] { 0.5f, 0.1f };
        weights.OutputWeights[1] = new[] { 0.2f, 0.3f };
        
        float[] logits = LinearHead.Project(hidden, weights, config);

        //Logits[0] = (1.0 * 0.5) + (2.0*0.2) + 0.1 = 0.5+0.4+0.1 = 1.0f
        //Logits[1] = (1.0 * 0.1) + (2.0 * 0.3) + 0.2 = 0.1 + 0.6 + 0.2 = 0.9f
        Assert.Multiple(() =>
        {
            Assert.That(logits.Length, Is.EqualTo(2));
            Assert.That(logits[0], Is.EqualTo(1.0f).Within(0.00001)); //Within is need to correct loss because of flaot
            Assert.That(logits[1], Is.EqualTo(0.9f).Within(0.00001));
        });
    }

    [Test]
    public void Project_CalculatesCorrectWeights()
    {
        var config = new TinyNNConfig(2,2,2);
        var weights = new TinyNNWeights(config);
        
        float[] hidden = { 1.0f, 2.0f };
        weights.OutputBias = new[] { 0.1f, 0.2f };
        weights.OutputWeights[0] = new[] { 0.5f, 0.1f };
        weights.OutputWeights[1] = new[] { 0.2f, 0.3f };
        
        float initialBias0 = weights.OutputBias[0];
        float initialWeight00 = weights.OutputWeights[0][0];
        
        float[] dLogits = { 0.5f, -0.2f };
        float lr = 0.1f;
        
        float[] dHidden = LinearHead.Backward(dLogits, hidden, weights, config, lr);
        
        Assert.Multiple(() =>
        {
            Assert.That(dHidden, Is.Not.Null);
            Assert.That(dHidden.Length, Is.EqualTo(config.EmbeddingSize));
            Assert.That(weights.OutputBias[0], Is.Not.EqualTo(initialBias0));
            Assert.That(weights.OutputWeights[0][0], Is.Not.EqualTo(initialWeight00));
        });
    }
    
    [Test]
    public void NextTokenScores_ContextLongerThanLimit_SlicesCorrectly()
    {
        var config = new TinyNNConfig(10,  4, 2); 
        var weights = new TinyNNWeights(config);
        var fakeMath = new MathOpsImpl();
        var model = new TinyNNModel(config, weights, fakeMath);

        int[] longContext = { 1, 2, 3, 4, 5 }; 
        int[] shortTail = { 4, 5 };            
        
        var longLogits = model.NextTokenScores(longContext);
        var shortLogits = model.NextTokenScores(shortTail);

        
        Assert.That(longLogits, Is.EqualTo(shortLogits));
    }
    
    [Test]
    public void NextTokenScores_EmptyContext_DoesNotCrash()
    {
        var config = new TinyNNConfig(10, 4, 3);
        var weights = new TinyNNWeights(config);
        var fakeMath = new MathOpsImpl();
        var model = new TinyNNModel(config, weights, fakeMath);

        Assert.DoesNotThrow(() => model.NextTokenScores(Array.Empty<int>()));
    }
    
    [Test]
    public void Test_ArgMax_FindsMaximum()
    {
        
        float[] scores = { 0.5f, 1.2f, 5.0f, 0.8f }; 
 
        int result = MathOps.Default.ArgMax(scores);

        Assert.That(result, Is.EqualTo(2)); 
    }

    [Test]
    public void Test_Softmax_SumIsOne()
    {
 
        float[] logits = { 1.0f, 2.0f, 3.0f };

        float[] probs = MathOps.Default.Softmax(logits);

        float sum = 0;
        foreach (var p in probs) sum += p;

        Assert.That(sum, Is.EqualTo(1.0f).Within(0.0001f));
    }

    [Test]
    public void Test_Loss_CorrectPrediction()
    {

        float[] logits = { 20.0f, 0.0f, 0.0f }; 
        int target = 0; 

        float loss = MathOps.Default.CrossEntropyLoss(logits, target);

        Assert.That(loss, Is.LessThan(0.001f));
    }

    [Test]
    public void Test_Sample_ReturnsValidIndex()
    {
        float[] probs = { 0.1f, 0.7f, 0.2f };
        Random rng = new Random();

        for (int i = 0; i < 100; i++)
        {
            int result = MathOps.Default.SampleFromProbs(probs, rng);

            Assert.That(result, Is.GreaterThanOrEqualTo(0));
            Assert.That(result, Is.LessThan(probs.Length));
        }
    }
    
    [TestCase(new[] { 0, 2, 1 })]
    [TestCase(new[] { 1, 1, 1 })]
    [TestCase(new[] { 1, 0, 2 })]
    public void Test_Softmax_SumIsOne(int[] context)
    {
        float[][] x = selfAttentionLayer.InitXmatrix(context);

        float[][] Q = selfAttentionLayer.InitMatrix(x, QKV.Q);
        float[][] K = selfAttentionLayer.InitMatrix(x, QKV.K);
        float[][] V = selfAttentionLayer.InitMatrix(x, QKV.V);

        float[][] scores = MatrixHelper.MultiplyMatrix(Q, MatrixHelper.TransposeMatrix(K));
        selfAttentionLayer.EachElementDivideBySquareRootOfEmbeddingSizeWithMask(scores);

        float[][] attn = selfAttentionLayer.SoftmaxEachRow(scores);

        for (int i = 0; i < attn.Length; i++)
        {
            float sum = 0f;
            for (int j = 0; j < attn[i].Length; j++)
            {
                sum += attn[i][j];
            }

            if (Math.Abs(sum - 1.0) < 1e-12)
            {
                sum = 1;
            }

            Assert.That(sum, Is.EqualTo(1.0f).Within(1e-6f));
        }
    }

    [TestCase(new[] { -0.123123f, 2.34827523f, -4.239785623f })]
    [TestCase(new[] { -0.000071562f, 0.00002836713f, -0.0000000623f })]
    [TestCase(new[] { -123123.123123f, 2.34827523f, -0.239785623f })]
    public void Test_Relu(float[] array)
    {
        FeedForwardLayer.Relu(array);
        
        for (int i = 0; i < array.Length; i++)
        {
            if (i % 2 == 0)
            {
                Assert.That(array[i], Is.EqualTo(0));
            }
            else
            {
                Assert.That(array[i], !Is.Negative);
            }
        }
    }
}