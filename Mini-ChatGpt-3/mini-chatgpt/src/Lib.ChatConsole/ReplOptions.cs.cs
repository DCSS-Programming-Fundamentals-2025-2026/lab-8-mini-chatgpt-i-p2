namespace MiniChatGPT.ChatConsole
{
    public class ReplOptions
    {
        public float Temperature { get; set; }
        public int TopK { get; set; }
        public int MaxTokens { get; set; }
        public int? Seed { get; set; }
        public bool IsRunning { get; set; }

        public ReplOptions()
        {
            Temperature = 0.5f;
            TopK = 10;
            MaxTokens = 100;
            Seed = null;
            IsRunning = true;
        }
    }
}
