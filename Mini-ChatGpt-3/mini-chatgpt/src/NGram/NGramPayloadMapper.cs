namespace NGram
{
    public class NGramPayloadMapper
    {
        public float[][] BigramProbs
        {
            get;
            set;
        }

        public System.Collections.Generic.List<NGram.TrigramEntry> TrigramProbs
        {
            get;
            set;
        }
    }
}