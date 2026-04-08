public interface ICorpusLoader
{
    CorpusClass Load(string path, CorpusLoadOptions options);
}