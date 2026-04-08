public class CorpusTextNormalizer
{
    public string Normalize(bool lowercase, string text)
    {
        if (text == null)
        {
            throw new NullReferenceException("Нічого нормалізувати");
        }

        if (lowercase == true)
        {
            text = text.ToLower();
        }

        return text;
    }
}