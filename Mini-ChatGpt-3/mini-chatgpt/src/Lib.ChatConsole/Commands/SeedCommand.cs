namespace MiniChatGPT.ChatConsole.Commands
{
    public class SeedCommand : IReplCommand
    {
        public string Name
        {
            get { return "/seed"; }
        }
        public string Description
        {
            get { return "Встановлює Seed для генерації"; }
        }

        public void Execute(string[] args, CommandExecutionContext context)
        {
            if (args.Length > 0 && int.TryParse(args[0], out int newSeed))
            {
                context.Options.Seed = newSeed;
                context.PrintMessage("\nЗначення Seed оновлено");

                context.PrintMessage(context.GetStatusInfo());
            }
            else
            {
                context.PrintMessage("Вказано некоректне число");
            }
        }
    }
}
