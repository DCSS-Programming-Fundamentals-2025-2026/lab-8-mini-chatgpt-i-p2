using Lib.Corpus.Configuration;

namespace Lib.Corpus.Infrastructure
{
    public class DefaultFileSystem : IFileSystem
    {
        public string ReadAllText(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                throw new FileNotFoundException("Couldn't find directory");
            }
            return File.ReadAllText(path);
        }

        public bool Exists(string path)
        {
            return File.Exists(path);
        }
    }
}