using Lib.Corpus.Configuration;
using Lib.Corpus.Domain;
using Lib.Corpus.Processing;

namespace Lib.Corpus.Tests;

public class Fake : IFileSystem
{
    public string? File { get; set; }
    public bool ExistsFile { get; set; }
    public string ReadAllText(string text) => File;
    public bool Exists(string text) => ExistsFile;
}

public class Tests
{
    private CorpusLoader loader;
    private Fake FakeFikeSystem;
    private CorpusLoadOptions loadOptions;
    private CorpusSplitter corpusSplitter;
    private CorpusTextNormalizer corpusTextNormalizer;

    [SetUp]
    public void SetUp()
    {
        FakeFikeSystem = new Fake();
        corpusSplitter = new CorpusSplitter();
        corpusTextNormalizer = new CorpusTextNormalizer();

        loader = new CorpusLoader(corpusTextNormalizer, corpusSplitter, FakeFikeSystem);
        loadOptions = new CorpusLoadOptions();
    }


    [Test]
    public void ExistsingAndSplitCheck()
    {
        // 10 символів, 10% валідації (1 символ)
        loadOptions.FallBack = "0123456789";

        CorpusClass corpus = loader.Load("not existing File", loadOptions);

        Assert.That(corpus, Is.Not.Null);
        Assert.That(corpus.TrainText, Is.EqualTo("012345678")); // 9 символів
        Assert.That(corpus.ValText, Is.EqualTo("9"));           // 1 символ
    }


    [Test]
    public void Check2_ExistingFile_LowerCaseFalse()
    {
        FakeFikeSystem.ExistsFile = true;
        FakeFikeSystem.File = "ABCDEFGHIJ"; // 10 символів у верхньому регістрі

        loadOptions.LowerCase = false;
        loadOptions.ValidateFraction = 0.2;

        CorpusClass corpus = loader.Load("notExisting.txt", loadOptions);

        Assert.That(corpus.TrainText, Is.EqualTo("ABCDEFGH"));  //8
        Assert.That(corpus.ValText, Is.EqualTo("IJ"));    //2
    }

    [Test]
    public void Check3_EmptyFileThrowsException()
    {
        FakeFikeSystem.ExistsFile = true;
        FakeFikeSystem.File = ""; // Порожній файл

        Assert.Throws<NullReferenceException>(() => loader.Load("empty.txt", loadOptions));
    }

    [Test]
    public void Load_OptionsAreNull_CreatesDefaultOptionsAndAppliesFallback()
    {
        FakeFikeSystem.ExistsFile = false; 

        CorpusClass corpus = loader.Load("missing.txt", null);

        Assert.That(corpus, Is.Not.Null);
        Assert.That(corpus.TrainText, Is.EqualTo("запасне значення"));
    }
}
        
        
        
















