namespace Lib.Tokenization.Model;

public class Vocabulary
{
    private readonly Dictionary<char, int> _charToId = new Dictionary<char, int>();
    private readonly Dictionary<int, char> _idToChar = new Dictionary<int, char>();

    public int Size 
    { 
        get { return _charToId.Count; } 
    }

    public const int UnkId = 0;
    public const char UnkChar = '\0'; 

    public Vocabulary()
    {
        _charToId[UnkChar] = UnkId;
        _idToChar[UnkId] = UnkChar;
    }

    public void BuildFromText(string text)
    {
        int nextId = Size; 
        foreach (char c in text)
        {
            if (_charToId.ContainsKey(c) == false)
            {
                _charToId[c] = nextId;
                _idToChar[nextId] = c;
                nextId++;
            }
        }
    }

    public int GetId(char c)
    {
        if (_charToId.ContainsKey(c))
        {
            return _charToId[c];
        }
        else
        {
            return UnkId;
        }
    }

    public char GetChar(int id)
    {
        if (_idToChar.ContainsKey(id))
        {
            return _idToChar[id];
        }
        else
        {
            return UnkChar;
        }
    }

    public List<string> GetPayload()
    {
        var list = new List<string>();
        for (int i = 0; i < Size; i++)
        {
            list.Add(GetChar(i).ToString());
        }
        return list;
    }

    public void LoadFromPayload(List<string> chars)
    {
        _charToId.Clear();
        _idToChar.Clear();
        for (int i = 0; i < chars.Count; i++)
        {
            char c;
            if (chars[i].Length > 0)
            {
                c = chars[i][0];
            }
            else
            {
                c = '\0';
            }

            _charToId[c] = i;
            _idToChar[i] = c;
        }
    }
}
