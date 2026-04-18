using System.Text;
using System.Text.RegularExpressions;
using Lib.Tokenization.Model;

namespace Lib.Tokenization;

public class WordTokenizer : ITokenizer
{
    private readonly WordVocabulary _vocabulary;
    private readonly string Version = "1.0.0";
    
    public string GetContractFingerprint()
    {
        return Version;
    }

    public WordTokenizer(WordVocabulary vocabulary)
    {
        _vocabulary = vocabulary;
    }

    public int VocabSize 
    { 
        get { return _vocabulary.Size; } 
    }

    public int[] Encode(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return new int[0];
        }

        string[] rawWords = Regex.Split(text, @"\s+");
        var tokens = new List<int>();

        foreach (string w in rawWords)
        {
            if (string.IsNullOrWhiteSpace(w))
            {
                continue;
            }
            string cleaned = WordVocabulary.CleanWord(w);
            tokens.Add(_vocabulary.GetId(cleaned));
        }

        return tokens.ToArray();
    }

    public string Decode(ReadOnlySpan<int> tokens)
    {
        if (tokens.Length == 0)
        {
            return "";
        }

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < tokens.Length; i++)
        {
            sb.Append(_vocabulary.GetWord(tokens[i]));
            if (i < tokens.Length - 1)
            {
                sb.Append(" ");
            }
        }
        return sb.ToString();
    }

    public object GetPayloadForCheckpoint()
    {
        return new { words = _vocabulary.GetPayload() };
    }

    public static WordTokenizer BuildFromText(string text)
    {
        WordVocabulary vocab = new WordVocabulary();
        vocab.BuildFromText(text);
        return new WordTokenizer(vocab);
    }
}
