public class CorpusLoader : ICorpusLoader
{
    private readonly CorpusTextNormalizer textNormalizer;
    private readonly CorpusSplitter corpusSplitter;
    private readonly IFileSystem defaultFileSystem;

    public CorpusLoader(CorpusTextNormalizer textNormalizer, CorpusSplitter corpusSplitter, IFileSystem defaultFileSystem)
    {
        this.textNormalizer = textNormalizer;
        this.corpusSplitter = corpusSplitter;
        this.defaultFileSystem = defaultFileSystem;
    }

    public CorpusClass Load(string path, CorpusLoadOptions? options)
    {
        if (options == null)
        {
            options = new CorpusLoadOptions();
        }

        bool exist = defaultFileSystem.Exists(path);
        string content;

        if (exist)
        {
            content = defaultFileSystem.ReadAllText(path);
        }
        else
        {
            content = options.FallBack ?? string.Empty;
        }

        return LoadFromText(content, options);
    }

    public static CorpusClass Load(string path)
    {
        CorpusLoader loader = new CorpusLoader(new CorpusTextNormalizer(), new CorpusSplitter(), new DefaultFileSystem());
        return loader.Load(path, null);
    }

    public CorpusClass LoadFromText(string text, CorpusLoadOptions? options)
    {
        if (options == null)
        {
            options = new CorpusLoadOptions();
        }

        text = textNormalizer.Normalize(options.LowerCase, text);

        string[] parts = corpusSplitter.Splitter(text, options.ValidateFraction);
        string TrainText = parts[0];
        string ValidatePart = parts[1];

        return new CorpusClass(TrainText, ValidatePart);
    }
}







