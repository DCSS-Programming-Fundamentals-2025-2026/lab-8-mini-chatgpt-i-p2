using CommandLine;

public class Options
{
    [Option("data", Default = "data/showcase.txt", HelpText = "Шлях до корпусу")]
    public string Data { get; set; }

    [Option("model", Default = "trigram", HelpText = "Тип моделі: bigram, trigram, tinynn, tinytransformer")]
    public string Model { get; set; }

    [Option("tokenizer", Default = "word", HelpText = "Токенізатор: word, char")]
    public string Tokenizer { get; set; }

    [Option("epochs", Default = 3, HelpText = "Кількість епох тренування")]
    public int Epochs { get; set; }

    [Option("out", Default = "data/checkpoint.json", HelpText = "Шлях для збереження checkpoint")]
    public string Out { get; set; }

    [Option("seed", Default = 42, HelpText = "Seed для відтворюваності")]
    public int Seed { get; set; }

    [Option("lr", Default = 0.1f, HelpText = "Learning rate (для TinyNN)")]
    public float LearningRate { get; set; }
}
