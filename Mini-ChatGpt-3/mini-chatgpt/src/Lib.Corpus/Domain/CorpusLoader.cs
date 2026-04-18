using Lib.Corpus.Configuration;
using Lib.Corpus.Processing;
using Lib.Corpus.Infrastructure;

namespace Lib.Corpus.Domain
{
    public class CorpusLoader : ICorpusLoader
    {
        private readonly CorpusTextNormalizer _textNormalizer;
        private readonly CorpusSplitter _corpusSplitter;
        private readonly IFileSystem _defaultFileSystem;
        private readonly string Version = "1.0.1";

        public CorpusLoader(CorpusTextNormalizer textNormalizer, CorpusSplitter corpusSplitter, IFileSystem defaultFileSystem)
        {
            this._textNormalizer = textNormalizer;
            this._corpusSplitter = corpusSplitter;
            this._defaultFileSystem = defaultFileSystem;
        }

        public static CorpusClass Load(string path)
        {
            CorpusLoader loader = new CorpusLoader(
                new CorpusTextNormalizer(),
                new CorpusSplitter(),
                new DefaultFileSystem()
            );

            return loader.Load(path, null);
        }

        public CorpusClass Load(string path, CorpusLoadOptions? options = null)
        {
            if (options == null)
            {
                options = new CorpusLoadOptions();
            }

            bool exist = this._defaultFileSystem.Exists(path);
            string content;

            if (exist)
            {
                content = this._defaultFileSystem.ReadAllText(path);
            }
            else
            {
                content = options.FallBack ?? string.Empty;
            }

            return this.LoadFromText(content, options);
        }

        public CorpusClass LoadFromText(string text, CorpusLoadOptions? options = null)
        {
            if (options == null)
            {
                options = new CorpusLoadOptions();
            }

            string normalizedText = this._textNormalizer.Normalize(options.LowerCase, text);

            string[] parts = this._corpusSplitter.Splitter(normalizedText, options.ValidateFraction);

            return new CorpusClass(parts[0], parts[1]);
        }

        public string GetContractFingerprint()
        {
            return this.Version;
        }
    }
}