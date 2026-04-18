namespace Lib.Corpus.Configuration
{
    public class CorpusLoadOptions
    {
        public bool LowerCase { get; set; } = true;
        public double ValidateFraction { get; set; } = 0.1;
        public string? FallBack { get; set; } = "Запасне значення ";

        public CorpusLoadOptions(bool lowerCase, double validateFraction, string fallBack)
        {
            this.LowerCase = lowerCase;
            this.ValidateFraction = validateFraction;
            this.FallBack = fallBack;
        }

        public CorpusLoadOptions() { }
    }
}
