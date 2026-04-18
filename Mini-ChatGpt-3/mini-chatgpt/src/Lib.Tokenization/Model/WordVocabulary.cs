using System.Text.RegularExpressions;
namespace Lib.Tokenization.Model;

public class WordVocabulary
{
    private readonly Dictionary<string, int> _wordToId = new Dictionary<string, int>();
    private readonly Dictionary<int, string> _idToWord = new Dictionary<int, string>();

    public int Size 
    { 
        get { return _wordToId.Count; } 
    }

    public const int UnkId = 0;
    public const string UnkWord = "<UNK>"; 

    public WordVocabulary()
    {
        _wordToId[UnkWord] = UnkId;
        _idToWord[UnkId] = UnkWord;
    }

    public void BuildFromText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return;
        }

        string[] rawWords = Regex.Split(text, @"\s+");
        int nextId = Size; 

        foreach (string w in rawWords)
        {
            if (string.IsNullOrWhiteSpace(w))
            {
                continue;
            }

            string cleanedWord = CleanWord(w);
            if (string.IsNullOrEmpty(cleanedWord) == false && _wordToId.ContainsKey(cleanedWord) == false)
            {
                _wordToId[cleanedWord] = nextId;
                _idToWord[nextId] = cleanedWord;
                nextId++;
            }
        }
    }

    public static string CleanWord(string word)
    {
        var cleaned = new System.Text.StringBuilder();
        foreach (char c in word)
        {
            if (char.IsPunctuation(c) == false) 
            {
                cleaned.Append(c);
            }
        }
        return cleaned.ToString().ToLowerInvariant(); 
    }

    public int GetId(string word)
    {
        if (_wordToId.ContainsKey(word))
        {
            return _wordToId[word];
        }
        else
        {
            return UnkId;
        }
    }

    public string GetWord(int id)
    {
        if (_idToWord.ContainsKey(id))
        {
            return _idToWord[id];
        }
        else
        {
            return UnkWord;
        }
    }

    public List<string> GetPayload()
    {
        var list = new List<string>();
        for (int i = 0; i < Size; i++)
        {
            list.Add(GetWord(i));
        }
        return list;
    }

    public void LoadFromPayload(List<string> words)
    {
        _wordToId.Clear();
        _idToWord.Clear();
        for (int i = 0; i < words.Count; i++)
        {
            string w = words[i];
            _wordToId[w] = i;
            _idToWord[i] = w;
        }
    }
}
