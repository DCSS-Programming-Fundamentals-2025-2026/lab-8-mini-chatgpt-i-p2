namespace Lib.Corpus.Configuration
{
    public interface IFileSystem
    {
        public string ReadAllText(string path);
        public bool Exists(string path);
    }
}