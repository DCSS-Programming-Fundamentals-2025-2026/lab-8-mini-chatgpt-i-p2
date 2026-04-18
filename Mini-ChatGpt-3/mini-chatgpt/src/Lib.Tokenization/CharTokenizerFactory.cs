using System.Text.Json;
using Lib.Tokenization.Model;
using Lib.Tokenization.Serialization;

namespace Lib.Tokenization;

public class CharTokenizerFactory : ITokenizerFactory
{
    public ITokenizer BuildFromText(string text)
    {
        var vocab = new Vocabulary();
        if (string.IsNullOrEmpty(text) == false)
        {
            vocab.BuildFromText(text);
        }
        return new CharTokenizer(vocab);
    }

    public ITokenizer FromPayload(JsonElement payload)
    {
        var vocab = TokenizerPayloadSerializer.DeserializeCharVocab(payload);
        return new CharTokenizer(vocab);
    }
}
