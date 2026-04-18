using MiniChatGPT.Contracts;
using MiniChatGPT.ChatConsole.Commands;

namespace MiniChatGPT.ChatConsole
{
    public class ChatRepl
    {
        private readonly ITextGenerator _generator;
        private readonly CommandRegistry _commandRegistry;
        private readonly ReplOptions _options;
        private readonly CommandExecutionContext _commandContext;

        public ChatRepl(ITextGenerator generator)
        {
            _generator = generator;
            _options = new ReplOptions();
            _commandRegistry = new CommandRegistry();

            PrintMessage printDelegate = (message) =>
            {
                Console.WriteLine($"[{GetTimestamp()}] Sys > {message}");
            };

            _commandContext = new CommandExecutionContext(_options, _generator, printDelegate);
        }

        public void Run(float temp, int topK)
        {
            _options.Temperature = temp;
            _options.TopK = topK;
            _options.Seed = null;
            _options.IsRunning = true;

            PrintBanner();
            _commandContext.PrintMessage(_commandContext.GetStatusInfo());
            _commandContext.PrintMessage("Введіть /help для списку команд.");
            Console.WriteLine();

            while (_options.IsRunning)
            {
                Console.Write($"[{GetTimestamp()}] You > ");
                var input = Console.ReadLine()?.Trim().ToLower();

                if (string.IsNullOrWhiteSpace(input))
                {
                    continue;
                }

                bool isCommand = _commandRegistry.TryExecute(input, _commandContext);

                if (!isCommand)
                {
                    try
                    {
                        var response = _generator.Generate(
                            input,
                            _options.MaxTokens,
                            _options.Temperature,
                            _options.TopK,
                            _options.Seed
                        );

                        Console.WriteLine($"[{GetTimestamp()}] Bot > {response}");
                    }
                    catch (Exception ex)
                    {
                        _commandContext.PrintMessage($"Помилка генерації: {ex.Message}");
                    }
                }

                Console.WriteLine();
            }

            _commandContext.PrintMessage("Сесію завершено...");
        }

        private string GetTimestamp()
        {
            return DateTime.Now.ToString("HH:mm");
        }

        private void PrintBanner()
        {
            Console.WriteLine("\nMini ChatGPT REPL");
        }
    }
}
