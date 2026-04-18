using Lib.Corpus.Processing;
using System.IO;

namespace Lib.Corpus.Tests;

public class SplitterTests
{
    private string content;
    private double validateFraction = 0.1;
    CorpusSplitter splitter = new();

    [SetUp]
    public void Setup()
    {
        content = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau up";
        validateFraction = 0.1;
    }

    [Test]
    public void Splitter_Success()
    {
        string expectedTrainText = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho si";
        string[] parts = splitter.Splitter(content, validateFraction);

        Assert.That(expectedTrainText, Is.EqualTo(parts[0].ToString()));
    }

    [Test]
    public void Splitter_Fail_TextIsNull()
    {
        content = null;

        Assert.Throws<NullReferenceException>(() => splitter.Splitter(content, validateFraction));
    }

    [Test]
    public void Splitter_Fail_ThrowsCorrectMessage()
    {
        content = null;
        string expectedMessage = "Nothing to split here";

        NullReferenceException ex = Assert.Throws<NullReferenceException>(() => splitter.Splitter(content, validateFraction));
        Assert.That(expectedMessage, Is.EqualTo(ex.Message));
    }

    [Test]
    public void Splitter_Success_LengthIsSame()
    {
        string[] parts = splitter.Splitter(content, validateFraction);

        int actualLength = parts[0].Length + parts[1].Length;
        int expectedLength = content.Length;

        Assert.That(expectedLength, Is.EqualTo(actualLength));
    }

    [Test]
    public void Splitter_Fail_NegativeFraction()
    {
        validateFraction = -1;

        Assert.Throws<Exception>(() => splitter.Splitter(content, validateFraction));
    }

    [Test]
    public void Splitter_Fail_NegativeFraction_ThrowsCorrectException()
    {
        validateFraction = -1;
        string expectedMessage = "Incorrect validate fraction.";

        Exception ex = Assert.Throws<Exception>(() => splitter.Splitter(content, validateFraction));
        Assert.That(expectedMessage, Is.EqualTo(ex.Message));
    }

    [Test]
    public void Splitter_Fail_MoreThanOne()
    {
        validateFraction = 1.1;

        Assert.Throws<Exception>(() => splitter.Splitter(content, validateFraction));
    }

    [Test]
    public void Splitter_Success_AllTextIsCorrect()
    {
        content = "1234567890";
        string expectedValText = "0";
        string expectedTrainText = "123456789";

        string[] parts = splitter.Splitter(content, validateFraction);

        Assert.That(expectedTrainText, Is.EqualTo(parts[0]));
        Assert.That(expectedValText, Is.EqualTo(parts[1]));
    }

    [Test]
    public void Spltter_Fail_SplittedTextToShort()
    {
        content = "abcdefg";

        Assert.Throws<Exception>(() => splitter.Splitter(content, validateFraction));
    }
}
