namespace MiniChatGPT.ChatConsole.Commands
{
    public class TopKCommand : IReplCommand
    {
        public string Name
        {
            get { return "/topk"; }
        }
        public string Description
        {
            get { return "Встановлює TopK для генерації"; }
        }

        public void Execute(string[] args, CommandExecutionContext context)
        {
            if (args.Length > 0 && int.TryParse(args[0], out int newTopK) && newTopK > 0)
            {
                context.Options.TopK = newTopK;
                context.PrintMessage("\nЗначення TopK оновлено");

                context.PrintMessage(context.GetStatusInfo());
            }
            else
            {
                context.PrintMessage("Вказано некоректне число (має бути ціле число більше за 0)");
            }
        }
    }
}
