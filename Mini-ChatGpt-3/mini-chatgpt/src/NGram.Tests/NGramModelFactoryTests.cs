using NUnit.Framework;
using NGram.ModelFactory;

namespace NGram.Tests;

public class NGramModelFactoryTests
{
    private const int VocabSize = 5;

    [Test]
    public void FactoryWorkingTest()
    {
        NGramModelFactory localFactory = new NGramModelFactory();

        NGramModel bi = localFactory.CreateBigramModel(VocabSize);
        TrigramModel tri = localFactory.CreateTrigramModel(VocabSize);

        Assert.That(bi, Is.Not.Null);
        Assert.That(tri, Is.Not.Null);
        Assert.That(bi, Is.TypeOf<NGramModel>());
        Assert.That(tri, Is.TypeOf<TrigramModel>());
    }
}
