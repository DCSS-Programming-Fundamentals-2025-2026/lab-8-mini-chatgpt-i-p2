namespace Lib.Corpus.Processing
{
    public class CorpusTextNormalizer
    {
        public string Normalize(bool lowercase, string text)
        {
            if (text == null)
            {
                throw new NullReferenceException("Nothing to normalize here");
            }

            if (lowercase == true)
            {
                text = text.ToLower();
            }

            return text;
        }
    }
}