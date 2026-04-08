public class CorpusSplitter
{
    public string[] Splitter(string text, double ValidateFraction)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            throw new NullReferenceException("Нічого розділяти");
        }

        int Validate = (int)(text.Length * ValidateFraction);

        if (ValidateFraction <= 0 || ValidateFraction >= 1 || Math.Truncate((decimal)Validate) == 0 || Math.Truncate((decimal)(text.Length - Validate)) == 0)
        {
            throw new Exception("Неправильна перевірка дробу");
        }

        int TrainPart = text.Length - Validate;

        string TrainText = text.Substring(0, TrainPart);
        string ValidatePart = text.Substring(TrainPart);

        string[] parts = { TrainText, ValidatePart };
        return parts;
    }
}


