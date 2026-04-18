using Lib.Corpus.Infrastructure;
using System.IO;

namespace Lib.Corpus.Tests;

public class DefaultFileSystemTests
{
    private string path;
    private string content;
    DefaultFileSystem fileSystem = new();

    [SetUp]
    public void Setup()
    {
        path = "testFile.txt";
        content = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau up";
        File.WriteAllText(path, content);
    }

    [Test]
    public void ReadAllText_Success()
    {
        string expectedText = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau up";
        string actualText = fileSystem.ReadAllText(path);
        Assert.That(expectedText, Is.EqualTo(actualText));
    }

    [Test]
    public void ReadAllText_PathIsIncorrect()
    {
        string pathNull = null;
        string pathEmpty = "";
        string pathWhiteSpace = " ";

        Assert.Throws<FileNotFoundException>(() => fileSystem.ReadAllText(pathNull));
        Assert.Throws<FileNotFoundException>(() => fileSystem.ReadAllText(pathEmpty));
        Assert.Throws<FileNotFoundException>(() => fileSystem.ReadAllText(pathWhiteSpace));
    }

    [Test]
    public void ReadAllText_PathIsIncorrect_ThrowsCorrectMessage()
    {
        path = "";
        string expectedMessage = "Couldn't find directory";

        FileNotFoundException ex = Assert.Throws<FileNotFoundException>(() => fileSystem.ReadAllText(path));
        Assert.That(expectedMessage, Is.EqualTo(ex.Message));
    }

    [Test]
    public void Exists_Success()
    {
        bool exists = fileSystem.Exists(path);
        Assert.That(exists, Is.True);
    }

    [Test]
    public void Exists_Fail_PathIsIncorrect()
    {
        path = "";
        bool exists = fileSystem.Exists(path);
        Assert.That(exists, Is.False);
    }
}





















