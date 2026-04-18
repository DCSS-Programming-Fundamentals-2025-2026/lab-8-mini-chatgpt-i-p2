using System.Text;
using Lib.Tokenization.Model;

namespace Lib.Tokenization;

public class CharTokenizer : ITokenizer
{
    private readonly Vocabulary _vocabulary;
    private readonly string Version = "1.0.0";
    
    public string GetContractFingerprint()
    {
        return Version;
    }

    public CharTokenizer(Vocabulary vocabulary)
    {
        _vocabulary = vocabulary;
    }

    public int VocabSize 
    { 
        get { return _vocabulary.Size; } 
    }

    public int[] Encode(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return new int[0];
        }

        int[] tokens = new int[text.Length];
        for (int i = 0; i < text.Length; i++)
        {
            tokens[i] = _vocabulary.GetId(text[i]);
        }
        return tokens;
    }

    public string Decode(ReadOnlySpan<int> tokens)
    {
        if (tokens.Length == 0)
        {
            return "";
        }

        StringBuilder sb = new StringBuilder();
        foreach (int token in tokens)
        {
            sb.Append(_vocabulary.GetChar(token));
        }
        return sb.ToString();
    }

    public object GetPayloadForCheckpoint()
    {
        return new { chars = _vocabulary.GetPayload() };
    }
}
