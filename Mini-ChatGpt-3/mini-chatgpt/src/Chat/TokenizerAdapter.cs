namespace MiniChatGPT.Chat
{
    public class TokenizerAdapter : MiniChatGPT.Contracts.ITokenizer
    {
        private readonly dynamic _originalTokenizer;

        public TokenizerAdapter(object originalTokenizer)
        {
            _originalTokenizer = originalTokenizer;
        }

        public int VocabSize => (int)_originalTokenizer.VocabSize;

        public int[] Encode(string text)
        {
            return (int[])_originalTokenizer.Encode(text);
        }

        public string Decode(int[] tokens)
        {
            string result = (string)_originalTokenizer.Decode(tokens);

            if (result.Contains("<EOS>"))
            {
                result = result.Split("<EOS>")[0];
            }

            return result.Replace("<UNK>", "").Trim();
        }

        public string Decode(ReadOnlySpan<int> tokens)
        {
            return Decode(tokens.ToArray());
        }

        public object GetPayloadForCheckpoint()
        {
            try
            {
                return _originalTokenizer.GetPayloadForCheckpoint();
            }
            catch
            {
                return null;
            }
        }

        public string GetContractFingerprint()
        {
            return "Adapter-B1-to-Contracts";
        }
    }
}
