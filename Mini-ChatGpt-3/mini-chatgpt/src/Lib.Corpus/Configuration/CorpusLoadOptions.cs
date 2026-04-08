public class CorpusLoadOptions
{
    public bool LowerCase { get; set; } = true;
    public double ValidateFraction { get; set; } = 0.1;
    public string? FallBack { get; set; } = "Запасне значення ";

    public CorpusLoadOptions(bool LowerCase, double ValidateFraction, string FallBack)
    {
        this.LowerCase = LowerCase;
        this.ValidateFraction = ValidateFraction;
        this.FallBack = FallBack;
    }

    public CorpusLoadOptions() { }
}