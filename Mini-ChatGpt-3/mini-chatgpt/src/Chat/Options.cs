using CommandLine;
namespace MiniChatGPT.Chat;

public class Options
{
    [Option("checkpoint", Default = "data/checkpoint.json", HelpText = "Чек-поінт")]
    public string Checkpoint { get; set; }

    [Option("temp", Default = 0.7f, HelpText = "Температура")]
    public float Temp { get; set; }

    [Option("topk", Default = 10, HelpText = "Кращий з ...")]
    public int TopK { get; set; }
}