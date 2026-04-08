namespace MiniChatGPT.ChatConsole.Commands
{
    public class ResetCommand : IReplCommand
    {
        public string Name
        {
            get { return "/reset"; }
        }
        public string Description
        {
            get { return "Ресетить чат"; }
        }

        public void Execute(string[] args, CommandExecutionContext context)
        {
            context.PrintMessage("Чат ресетнувся");
        }
    }
}
