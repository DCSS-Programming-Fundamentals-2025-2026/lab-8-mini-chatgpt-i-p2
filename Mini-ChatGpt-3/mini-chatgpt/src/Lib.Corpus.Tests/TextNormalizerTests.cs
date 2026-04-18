using Lib.Corpus.Processing;

namespace Lib.Corpus.Tests;

public class TextNormalizerTests
{
    [Test]
    public void Normalize_Success()
    {
        CorpusTextNormalizer textNormalizer = new();
        bool isTrue = true;
        string textToNormalize = "ABCDEFG";

        string normalizedText = textNormalizer.Normalize(isTrue, textToNormalize);

        Assert.That(textToNormalize.ToLower(), Is.EqualTo(normalizedText));
    }

    [Test]
    public void Normalize_Fail_BoolConditionIsFalse()
    {
        CorpusTextNormalizer textNormalizer = new();
        bool isTrue = false;
        string textToNormalize = "ABCDEFG";

        string normalizedText = textNormalizer.Normalize(isTrue, textToNormalize);

        Assert.That(textToNormalize, Is.EqualTo(normalizedText));
    }

    [Test]
    public void Normalize_Fail_TextIsEmpty()
    {
        CorpusTextNormalizer textNormalizer = new();
        bool isTrue = true;
        string textToNormalizeEmpty = "";

        string normalizedTextEmpty = textNormalizer.Normalize(isTrue, textToNormalizeEmpty);

        Assert.That(textToNormalizeEmpty.ToLower(), Is.EqualTo(normalizedTextEmpty));
    }

    [Test]
    public void Normalize_Fail_TextIsNull()
    {
        CorpusTextNormalizer textNormalizer = new();
        bool isTrue = true;
        string textToNormalize = null;

        Assert.Throws<NullReferenceException>(() => textNormalizer.Normalize(isTrue, textToNormalize));
    }

    [Test]
    public void Normalize_FailWithCorrectMessage()
    {
        CorpusTextNormalizer textNormalizer = new();
        bool isTrue = true;
        string textToNormalize = null;
        string expectedMessege = "Nothing to normalize here";

        NullReferenceException ex = Assert.Throws<NullReferenceException>(() => textNormalizer.Normalize(isTrue, textToNormalize));

        Assert.That(expectedMessege, Is.EqualTo(ex.Message));
    }
}
