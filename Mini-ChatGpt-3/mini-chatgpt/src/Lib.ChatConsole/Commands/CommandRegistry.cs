namespace MiniChatGPT.ChatConsole.Commands
{
    public class CommandRegistry
    {
        private readonly Dictionary<string, IReplCommand> _commands;
        public Func<IEnumerable<IReplCommand>> GetCommands { get; }

        public CommandRegistry()
        {
            _commands = new Dictionary<string, IReplCommand>();

            GetCommands = () => { return _commands.Values; };

            Register(new HelpCommand(GetCommands));
            Register(new ResetCommand());
            Register(new SeedCommand());
            Register(new TempCommand());
            Register(new TopKCommand());
            Register(new QuitCommand());
        }

        public void Register(IReplCommand command)
        {
            _commands[command.Name] = command;
        }

        public bool TryExecute(string input, CommandExecutionContext context)
        {
            if (string.IsNullOrWhiteSpace(input) || !input.StartsWith("/"))
            {
                return false;
            }

            var parts = input.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var commandName = parts[0].ToLowerInvariant();
            var args = parts.Skip(1).ToArray();

            if (_commands.TryGetValue(commandName, out var command))
            {
                command.Execute(args, context);
                return true;
            }
            else
            {
                context.PrintMessage($"Невідома команда: {commandName}. Введіть /help для списку команд.");

                return true;
            }
        }
    }
}
