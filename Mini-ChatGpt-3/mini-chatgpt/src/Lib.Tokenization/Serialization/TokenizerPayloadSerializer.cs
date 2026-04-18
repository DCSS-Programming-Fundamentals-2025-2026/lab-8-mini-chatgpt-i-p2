using System.Text.Json;
using Lib.Tokenization.Model;

namespace Lib.Tokenization.Serialization;

public static class TokenizerPayloadSerializer
{
    public static Vocabulary DeserializeCharVocab(JsonElement payload)
    {
        var vocab = new Vocabulary();
        
        JsonElement charsElement;
        bool hasProperty = payload.TryGetProperty("chars", out charsElement);

        if (hasProperty && charsElement.ValueKind == JsonValueKind.Array)
        {
            var charsList = new List<string>();
            foreach (JsonElement item in charsElement.EnumerateArray())
            {
                string text = item.GetString();
                if (text == null)
                {
                    text = "";
                }
                charsList.Add(text);
            }
            vocab.LoadFromPayload(charsList);
        }
        return vocab;
    }
    
    public static WordVocabulary DeserializeWordVocab(JsonElement payload)
    {
        var vocab = new WordVocabulary();
        
        JsonElement wordsElement;
        bool hasProperty = payload.TryGetProperty("words", out wordsElement);

        if (hasProperty && wordsElement.ValueKind == JsonValueKind.Array)
        {
            var wordsList = new List<string>();
            foreach (JsonElement item in wordsElement.EnumerateArray())
            {
                string text = item.GetString();
                if (text == null)
                {
                    text = "";
                }
                wordsList.Add(text);
            }
            vocab.LoadFromPayload(wordsList);
        }
        return vocab;
    }
}
