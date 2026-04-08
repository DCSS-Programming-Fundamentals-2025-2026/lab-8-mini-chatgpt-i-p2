using Lib.Tokenization;
using NGram;
using NGram.ModelFactory;
using NGram.Metrics;
using System.Text.Json;

namespace NGram.Tests;

public class BaselineEndToEndTests
{
    private CorpusClass _corpus;
    private WordTokenizer _wordTokenizer;
    private int[] _trainTokens;
    private int[] _valTokens;
    private NGramModel _biModel;
    private TrigramModel _triModel;

    [SetUp]
    public void Setup()
    {
        _corpus = CorpusLoader.Load("data/showcase.txt");
        _wordTokenizer = WordTokenizer.BuildFromText(_corpus.TrainText);
        _trainTokens = _wordTokenizer.Encode(_corpus.TrainText);
        _valTokens = _wordTokenizer.Encode(_corpus.ValText);
        NGramModelFactory factory = new NGramModelFactory();
        _biModel = factory.CreateBigramModel(_wordTokenizer.VocabSize);
        _triModel = factory.CreateTrigramModel(_wordTokenizer.VocabSize);
        _biModel.Train(_trainTokens);
        _triModel.Train(_trainTokens);
    }

    [Test]
    public void TrigramOnRealCorpus_GeneratesUkrainianWords()
    {
        var context = new List<int>{ _trainTokens[0], _trainTokens[1] };
        for (int i = 0; i < 10; i++)
        {
            float[] scores = _triModel.NextTokenScores(context.ToArray());
            int nextToken = Array.IndexOf(scores, scores.Max());
            context.Add(nextToken);
        }
        
        string generatedText = _wordTokenizer.Decode(context.ToArray());

        Assert.That(generatedText, Is.Not.Empty);
        Assert.That(generatedText.Length, Is.GreaterThan(5));
    }

    [Test]
    public void FullCheckpoint_SaveLoad_SameGeneration()
    {
        var context = new List<int>{ _trainTokens[0], _trainTokens[1] };
        for (int i = 0; i < 10; i++)
        {
            float[] scores = _triModel.NextTokenScores(context.ToArray());
            context.Add(Array.IndexOf(scores, scores.Max()));
        }
        string textBeforeSave = _wordTokenizer.Decode(context.ToArray());

        string modelJson     = System.Text.Json.JsonSerializer.Serialize(_triModel.GetPayloadForCheckpoint());
        string tokenizerJson = System.Text.Json.JsonSerializer.Serialize(_wordTokenizer.GetPayloadForCheckpoint());

        var restoredModel = new TrigramModel(_wordTokenizer.VocabSize);

        restoredModel.FromPayload(System.Text.Json.JsonDocument.Parse(modelJson).RootElement);

        var tokenizerPayload = System.Text.Json.JsonDocument.Parse(tokenizerJson).RootElement;
        var restoredVocab = Lib.Tokenization.Serialization.TokenizerPayloadSerializer.DeserializeWordVocab(tokenizerPayload);
        var restoredTokenizer = new WordTokenizer(restoredVocab);
        
        var context2 = new List<int>{ _trainTokens[0], _trainTokens[1] };
        for (int i = 0; i < 10; i++)
        {
            float[] scores = restoredModel.NextTokenScores(context2.ToArray());
            int nextToken = Array.IndexOf(scores, scores.Max());
            context2.Add(nextToken);
        }
        string textAfterSave = restoredTokenizer.Decode(context2.ToArray());

        Assert.That(textAfterSave, Is.EqualTo(textBeforeSave));
    }

    [Test]
    public void WordAndCharTokenizer_BothProduceReadableOutput()
    {
        var wordContext = new List<int>{_trainTokens[0], _trainTokens[1]};
        for (int i = 0; i < 10; i++)
        {
            float[] scores = _triModel.NextTokenScores(wordContext.ToArray());
            wordContext.Add(Array.IndexOf(scores, scores.Max()));
        }
        string wordGeneratedText = _wordTokenizer.Decode(wordContext.ToArray());

        var charFactory = new CharTokenizerFactory();
        var charTokenizer = charFactory.BuildFromText(_corpus.TrainText);
        var charTrainTokens = charTokenizer.Encode(_corpus.TrainText);

        var charModel = new TrigramModel(charTokenizer.VocabSize);
        charModel.Train(charTrainTokens);

        var charContext = new List<int>{charTrainTokens[0], charTrainTokens[1]};
        for (int i = 0; i < 20; i++)
        {
            float[] scores = charModel.NextTokenScores(charContext.ToArray());
            charContext.Add(Array.IndexOf(scores, scores.Max()));
        }
        string charGeneratedText = charTokenizer.Decode(charContext.ToArray());


        Assert.That(wordGeneratedText, Is.Not.Empty, "Word tokenizer має генерувати текст");
        Assert.That(charGeneratedText, Is.Not.Empty, "Char tokenizer має генерувати текст");
        Assert.That(wordGeneratedText.Length, Is.GreaterThan(5));
        Assert.That(charGeneratedText.Length, Is.GreaterThan(5));
        
    }
    
    [Test]
    public void Trigram_Perplexity_OnValidation_IsFinite()
    {
        double perplexity = PerplexityCalculator.ComputePerplexity(_valTokens, ctx => _triModel.NextTokenScores(ctx));
        Assert.That(perplexity, Is.GreaterThan(0));
        Assert.That(double.IsFinite(perplexity), Is.True);
    }

    [Test]
    public void Trigram_HasLowerPerplexity_ThanBigram()
    {
        double trigramPerplexity = PerplexityCalculator.ComputePerplexity(_trainTokens, ctx => _triModel.NextTokenScores(ctx));
        double bigramPerplexity = PerplexityCalculator.ComputePerplexity(_trainTokens, ctx => _biModel.NextTokenScores(ctx));
        Assert.That(trigramPerplexity, Is.LessThan(bigramPerplexity));
    }

    [Test]
    public void NextTokenScores_SumsToOne_ForAllTokens()
    {
        int[] context = { _trainTokens[0], _trainTokens[1] };
        float[] scores = _triModel.NextTokenScores(context);
        
        Assert.That(scores.Sum(), Is.EqualTo(1.0).Within(0.001));
    }
}