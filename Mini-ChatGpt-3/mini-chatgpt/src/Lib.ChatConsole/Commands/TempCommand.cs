namespace MiniChatGPT.ChatConsole.Commands
{
    public class TempCommand : IReplCommand
    {
        public string Name
        {
            get { return "/temp"; }
        }
        public string Description
        {
            get { return "Встановлює температуру для генерації"; }
        }

        public void Execute(string[] args, CommandExecutionContext context)
        {
            if (args.Length > 0 && float.TryParse(args[0], out float newTemp) && newTemp >= 0)
            {
                context.Options.Temperature = newTemp;
                context.PrintMessage("\nЗначення температури оновлено");

                context.PrintMessage(context.GetStatusInfo());
            }
            else
            {
                context.PrintMessage("Вказано некоректне число (має бути 0 або більше)");
            }
        }
    }
}
