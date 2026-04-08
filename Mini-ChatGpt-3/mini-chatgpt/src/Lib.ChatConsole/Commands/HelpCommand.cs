namespace MiniChatGPT.ChatConsole.Commands
{
    public class HelpCommand : IReplCommand
    {
        public readonly Func<IEnumerable<IReplCommand>> GetCommnads;

        public HelpCommand(Func<IEnumerable<IReplCommand>> getCommnads)
        {
            GetCommnads = getCommnads;
        }

        public string Name
        {
            get { return "/help"; }
        }
        public string Description
        {
            get { return "Показує список доступних команд та їхні функції"; }
        }

        public void Execute(string[] args, CommandExecutionContext context)
        {
            context.PrintMessage("\nДоступні команди:");

            IEnumerable<IReplCommand> commnads = GetCommnads();

            foreach(var c in commnads)
            {
                context.PrintMessage($"{c.Name} - {c.Description}");
            }

            context.PrintMessage("");
        }
    }
}
