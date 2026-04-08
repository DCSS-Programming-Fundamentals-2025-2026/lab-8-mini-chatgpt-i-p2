using MiniChatGPT.Contracts;
using MiniChatGPT.ChatConsole.Commands;
using MiniChatGPT.ChatConsole;

namespace Lib.ChatConsole.Tests
{
    public class FakeTextGenerator : ITextGenerator
    {
        public string Generate(string prompt, int maxTokens, float temperature, int topK, int? seed = null)
        {
            return "Smth";
        }
    }

    public class ChatConsoleTests
    {
        private ReplOptions _options;
        private FakeTextGenerator _fakeGenerator;
        private List<string> _printedMessages;
        private CommandExecutionContext _context;
        private CommandRegistry _registry;

        [SetUp]
        public void SetUp()
        {
            _options = new ReplOptions();
            _fakeGenerator = new FakeTextGenerator();
            _printedMessages = new List<string>();

            PrintMessage capturedPrint = (message) =>
            {
                _printedMessages.Add(message);
            };

            _context = new CommandExecutionContext(_options, _fakeGenerator, capturedPrint);
            _registry = new CommandRegistry();
        }

        [Test]
        public void ValidInput_ChangesTemperatureAndPrintsStatus()
        {
            _registry.TryExecute("/temp 5", _context);

            Assert.That(_options.Temperature, Is.EqualTo(5f));
            Assert.That(_printedMessages.Count, Is.GreaterThanOrEqualTo(2));
            Assert.That(_printedMessages[0], Does.Contain("оновлено"));
        }

        [Test]
        public void TempCommand_InvalidNumber_PrintsError()
        {
            _registry.TryExecute("/temp aaa", _context);

            Assert.That(_options.Temperature, Is.EqualTo(0.5f));
            Assert.That(_printedMessages[0], Does.Contain("некоректне число"));
        }

        [Test]
        public void ValidInput_SetsIsRunningToFalse()
        {
            _options.IsRunning = true;

            _registry.TryExecute("/quit", _context);

            Assert.That(_options.IsRunning, Is.False);
            Assert.That(_printedMessages[0], Does.Contain("«авершенн€ роботи"));
        }

        [Test]
        public void TryExecute_UnknownCommand_ReturnsTrueAndPrintsError()
        {
            bool isCommand = _registry.TryExecute("/smth", _context);

            Assert.That(isCommand, Is.True);
            Assert.That(_printedMessages[0], Does.Contain("Ќев≥дома команда"));
        }

        [Test]
        public void TryExecute_NotACommand_ReturnsFalse()
        {
            bool isCommand = _registry.TryExecute("ѕрив≥т", _context);

            Assert.That(isCommand, Is.False);
            Assert.That(_printedMessages.Count, Is.EqualTo(0));
        }

        [Test]
        public void TryExecute_EmptyString_ReturnsFalse()
        {
            bool isCommand = _registry.TryExecute(" ", _context);

            Assert.That(isCommand, Is.False);
        }

        [Test]
        public void TopKCommand_ValidInteger_ChangesTopKAndPrintsStatus()
        {
            _registry.TryExecute("/topk 5", _context);

            Assert.That(_options.TopK, Is.EqualTo(5));
            Assert.That(_printedMessages.Count, Is.GreaterThanOrEqualTo(2));
            Assert.That(_printedMessages[0], Does.Contain("оновлено"));
        }

        [Test]
        public void TopKCommand_InvalidString_PrintsError()
        {
            _registry.TryExecute("/topk abc", _context);

            Assert.That(_options.TopK, Is.EqualTo(10));
            Assert.That(_printedMessages[0], Does.Contain("некоректне число"));
        }

        [Test]
        public void TopKCommand_FloatInsteadOfInteger_PrintsError()
        {
            _registry.TryExecute("/topk 5.5", _context);

            Assert.That(_options.TopK, Is.EqualTo(10));
            Assert.That(_printedMessages[0], Does.Contain("некоректне число"));
        }

        [Test]
        public void SeedCommand_ValidInteger_SetsSeedAndPrintsStatus()
        {
            _registry.TryExecute("/seed 42", _context);

            Assert.That(_options.Seed, Is.EqualTo(42));
            Assert.That(_printedMessages[0], Does.Contain("оновлено"));
        }

        [Test]
        public void SeedCommand_NoArguments_PrintsError()
        {
            _registry.TryExecute("/seed", _context);

            Assert.That(_options.Seed, Is.Null);
            Assert.That(_printedMessages[0], Does.Contain("некоректне число"));
        }

        [Test]
        public void ResetCommand_Always_PrintsResetMessage()
        {
            _registry.TryExecute("/reset", _context);

            Assert.That(_printedMessages.Count, Is.EqualTo(1));
            Assert.That(_printedMessages[0], Does.Contain("чат ресетнувс€").IgnoreCase.Or.Contain("ресетнувс€").IgnoreCase);
        }

        [Test]
        public void HelpCommand_Execute_OutputContainsQuitCommand()
        {
            string command = "/help";

            _registry.TryExecute(command, _context);

            Assert.That(_printedMessages, Has.Some.Contains("/quit"));
        }

        [Test]
        public void HelpCommand_Execute_OutputContainsTempCommand()
        {
            string command = "/help";

            _registry.TryExecute(command, _context);

            Assert.That(_printedMessages, Has.Some.Contains("/temp"));
        }

        [Test]
        public void TopKCommand_NegativeInteger_PrintsError()
        {
            _registry.TryExecute("/topk -10", _context);

            Assert.That(_options.TopK, Is.EqualTo(10)); 
            Assert.That(_printedMessages[0], Does.Contain("некоректне число"));
        }

        [Test]
        public void TempCommand_NegativeNumber_PrintsError()
        {
            _registry.TryExecute("/temp -1.5", _context);

            Assert.That(_options.Temperature, Is.EqualTo(0.5f)); 
            Assert.That(_printedMessages[0], Does.Contain("некоректне число"));
        }

        [Test]
        public void TempCommand_Zero_IsValid()
        {
            _registry.TryExecute("/temp 0", _context);

            Assert.That(_options.Temperature, Is.EqualTo(0f));
            Assert.That(_printedMessages[0], Does.Contain("оновлено"));
        }
    }
}