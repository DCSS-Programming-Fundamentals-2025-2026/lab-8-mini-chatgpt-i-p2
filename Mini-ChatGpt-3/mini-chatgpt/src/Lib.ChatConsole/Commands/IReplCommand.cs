namespace MiniChatGPT.ChatConsole.Commands
{
    public interface IReplCommand
    {
        public string Name { get; }
        public string Description { get; }

        public void Execute(string[] args, CommandExecutionContext context);
    }
}
