using NUnit.Framework;
using NGram.ModelFactory;
using System;
using System.Text.Json;

namespace NGram.Tests;

public class TrigramModelTests
{
    private TrigramModel _triModel;
    private const int VocabSize = 5;

    [SetUp]
    public void Setup()
    {
        NGramModelFactory factory = new NGramModelFactory();
        _triModel = factory.CreateTrigramModel(VocabSize);
    }

    [Test]
    public void TrigramModelTest()
    {
        int[] trainingData = new int[] { 1, 2, 3, 1, 2, 4, 1, 2, 3 };
        ReadOnlySpan<int> tokens = new ReadOnlySpan<int>(trainingData);

        _triModel.Train(tokens);

        _triModel.Train(tokens);
        int[] context = new int[] { 1, 2 };
        float[] scores = _triModel.NextTokenScores(new ReadOnlySpan<int>(context));

        Assert.That(scores.Length, Is.EqualTo(VocabSize));
        Assert.That(scores[3], Is.EqualTo(0.666f).Within(0.01f));
        Assert.That(scores[4], Is.EqualTo(0.333f).Within(0.01f));
    }

    [Test]
    public void TrigramEmptyContextTest()
    {
        int[] context = new int[] { };
        float[] scores = _triModel.NextTokenScores(new ReadOnlySpan<int>(context));

        Assert.That(scores.Length, Is.EqualTo(VocabSize));
        Assert.That(scores[0], Is.EqualTo(0.2f));
        Assert.That(scores[1], Is.EqualTo(0.2f));
        Assert.That(scores[2], Is.EqualTo(0.2f));
        Assert.That(scores[3], Is.EqualTo(0.2f));
        Assert.That(scores[4], Is.EqualTo(0.2f));
    }

    [Test]
    public void TrigramModel_UnknownTrigramPrefix_FallsBackToBigram()
    {
        int[] trainingData = new int[] { 1, 2, 3, 1, 2, 4, 0, 2, 3 };
        _triModel.Train(new ReadOnlySpan<int>(trainingData));

        int[] context = new int[] { 4, 2 };
        float[] scores = _triModel.NextTokenScores(new ReadOnlySpan<int>(context));

        Assert.That(scores.Length, Is.EqualTo(VocabSize));
        Assert.That(scores[3], Is.EqualTo(0.666f).Within(0.01f));
        Assert.That(scores[4], Is.EqualTo(0.333f).Within(0.01f));
    }

    [Test]
    public void TrigramModel_SerializationRoundTrip_Test()
    {
        int[] trainingData = new int[] { 1, 2, 3, 1, 2, 4, 1, 2, 3 };
        _triModel.Train(new ReadOnlySpan<int>(trainingData));

        object payloadObj = _triModel.GetPayloadForCheckpoint();
        string json = JsonSerializer.Serialize(payloadObj);
        JsonElement jsonElement = JsonSerializer.Deserialize<JsonElement>(json);

        NGramModelFactory factory = new NGramModelFactory();
        TrigramModel newTriModel = factory.CreateTrigramModel(VocabSize);
        newTriModel.FromPayload(jsonElement);

        int[] context = new int[] { 1, 2 };
        float[] originalScores = _triModel.NextTokenScores(new ReadOnlySpan<int>(context));
        float[] newScores = newTriModel.NextTokenScores(new ReadOnlySpan<int>(context));

        Assert.That(newScores.Length, Is.EqualTo(originalScores.Length));
        for (int i = 0; i < originalScores.Length; i++)
        {
            Assert.That(newScores[i], Is.EqualTo(originalScores[i]).Within(0.0001f));
        }
    }
}
