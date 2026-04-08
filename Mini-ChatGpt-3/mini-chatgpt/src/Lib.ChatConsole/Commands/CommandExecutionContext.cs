using MiniChatGPT.Contracts;

namespace MiniChatGPT.ChatConsole.Commands
{
    public delegate void PrintMessage(string message);
    public class CommandExecutionContext
    {
        public ReplOptions Options { get; }
        public ITextGenerator Generator { get; }
        public PrintMessage PrintMessage { get; }

        public Func<string> GetStatusInfo { get; }

        public CommandExecutionContext(ReplOptions options, ITextGenerator generator, PrintMessage printMessage)
        {
            Options = options;
            Generator = generator;
            PrintMessage = printMessage;

            GetStatusInfo = () => 
            {
                return $"[Статус] Temp: {Options.Temperature}, TopK: {Options.TopK}, Seed: {Options.Seed?.ToString() ?? "Порожній"}";
            };     
        }
    }
}
