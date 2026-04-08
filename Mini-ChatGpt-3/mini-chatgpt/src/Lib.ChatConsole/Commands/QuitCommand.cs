namespace MiniChatGPT.ChatConsole.Commands
{
    public class QuitCommand : IReplCommand
    {
        public string Name
        {
            get { return "/quit"; }
        }
        public string Description
        {
            get { return "Завершує роботу"; }
        }

        public void Execute(string[] args, CommandExecutionContext context)
        {
            context.Options.IsRunning = false;
            context.PrintMessage("\nЗавершення роботи");
        }
    }
}
