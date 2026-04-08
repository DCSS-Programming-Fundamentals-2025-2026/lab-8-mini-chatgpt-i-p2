using NUnit.Framework;
using NGram.ModelFactory;
using System;
using System.Text.Json;

namespace NGram.Tests;

public class BigramModelTests
{
    private NGramModel _biModel;
    private const int VocabSize = 5;

    [SetUp]
    public void Setup()
    {
        NGramModelFactory factory = new NGramModelFactory();
        _biModel = factory.CreateBigramModel(VocabSize);
    }

    [Test]
    public void BigramModelTest()
    {
        int[] trainingData = new int[] { 0, 1, 2, 1, 3 };
        ReadOnlySpan<int> tokens = new ReadOnlySpan<int>(trainingData);
        _biModel.Train(tokens);

        int[] context = new int[] { 0, 1 };
        float[] scores = _biModel.NextTokenScores(new ReadOnlySpan<int>(context));

        Assert.That(scores.Length, Is.EqualTo(VocabSize));
        Assert.That(scores[2], Is.EqualTo(0.5f));
        Assert.That(scores[3], Is.EqualTo(0.5f));
    }

    [Test]
    public void EmptyContextTest()
    {
        int[] context = new int[] { };
        float[] scores = _biModel.NextTokenScores(new ReadOnlySpan<int>(context));

        Assert.That(scores.Length, Is.EqualTo(VocabSize));
        Assert.That(scores[0], Is.EqualTo(0.2f));
        Assert.That(scores[1], Is.EqualTo(0.2f));
        Assert.That(scores[2], Is.EqualTo(0.2f));
        Assert.That(scores[3], Is.EqualTo(0.2f));
        Assert.That(scores[4], Is.EqualTo(0.2f));
    }

    [Test]
    public void BigramModel_LongContext_UsesOnlyLastToken()
    {
        int[] trainingData = new int[] { 0, 1, 2, 1, 3 };
        _biModel.Train(new ReadOnlySpan<int>(trainingData));

        int[] longContext = new int[] { 4, 3, 2, 4, 1 }; 
        float[] scores = _biModel.NextTokenScores(new ReadOnlySpan<int>(longContext));

        Assert.That(scores.Length, Is.EqualTo(VocabSize));
        Assert.That(scores[2], Is.EqualTo(0.5f));
        Assert.That(scores[3], Is.EqualTo(0.5f));
    }

    [Test]
    public void BigramModel_SerializationRoundTrip_Test()
    {
        int[] trainingData = new int[] { 0, 1, 2, 1, 3 };
        _biModel.Train(new ReadOnlySpan<int>(trainingData));

        object payloadObj = _biModel.GetPayloadForCheckpoint();
        string json = JsonSerializer.Serialize(payloadObj);
        JsonElement jsonElement = JsonSerializer.Deserialize<JsonElement>(json);

        NGramModelFactory factory = new NGramModelFactory();
        NGramModel newBiModel = factory.CreateBigramModel(VocabSize);
        newBiModel.FromPayload(jsonElement);

        int[] context = new int[] { 1 };
        float[] originalScores = _biModel.NextTokenScores(new ReadOnlySpan<int>(context));
        float[] newScores = newBiModel.NextTokenScores(new ReadOnlySpan<int>(context));

        Assert.That(newScores.Length, Is.EqualTo(originalScores.Length));
        for (int i = 0; i < originalScores.Length; i++)
        {
            Assert.That(newScores[i], Is.EqualTo(originalScores[i]).Within(0.0001f));
        }
    }
}
