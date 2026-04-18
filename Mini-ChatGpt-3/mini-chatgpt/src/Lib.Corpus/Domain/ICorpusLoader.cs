using Lib.Corpus.Configuration;

namespace Lib.Corpus.Domain
{
    public interface ICorpusLoader
    {
        CorpusClass Load(string path, CorpusLoadOptions options);
    }
}
