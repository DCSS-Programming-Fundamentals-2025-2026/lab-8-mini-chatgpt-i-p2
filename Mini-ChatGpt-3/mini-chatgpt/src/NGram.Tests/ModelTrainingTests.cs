using NUnit.Framework;
using NGram.ModelFactory;
using System;

namespace NGram.Tests;

public class ModelTrainingTests
{
    private NGramModel _biModel;
    private TrigramModel _triModel;
    private const int VocabSize = 5;

    [SetUp]
    public void Setup()
    {
        NGramModelFactory factory = new NGramModelFactory();
        _biModel = factory.CreateBigramModel(VocabSize);
        _triModel = factory.CreateTrigramModel(VocabSize);
    }

    [Test]
    public void TrainEmptyDataDoesNotThrow()
    {
        int[] emptyData = Array.Empty<int>();
        Assert.DoesNotThrow(() => _biModel.Train(emptyData));
        Assert.DoesNotThrow(() => _triModel.Train(emptyData));
    }

    [Test]
    public void TrainInsufficientDataDoesNotThrow()
    {
        int[] singleTokenData = new int[] { 1 };
        Assert.DoesNotThrow(() => _biModel.Train(singleTokenData));

        int[] twoTokensData = new int[] { 1, 2 };
        Assert.DoesNotThrow(() => _triModel.Train(twoTokensData));
    }
}
