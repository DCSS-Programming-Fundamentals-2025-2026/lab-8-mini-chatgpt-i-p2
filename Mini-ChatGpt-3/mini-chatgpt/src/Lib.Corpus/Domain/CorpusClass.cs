public record CorpusClass
{
    public string? TrainText { get; }
    public string? ValText { get; }

    public CorpusClass(string TrainText, string ValText)
    {
        this.TrainText = TrainText;
        this.ValText = ValText;
    }
}