using System.Text.Json;
using Lib.Tokenization.Model;
using Lib.Tokenization.Serialization;

namespace Lib.Tokenization;

public class WordTokenizerFactory : ITokenizerFactory
{
    public ITokenizer BuildFromText(string text)
    {
        return WordTokenizer.BuildFromText(text);
    }

    public ITokenizer FromPayload(JsonElement payload)
    {
        WordVocabulary vocab = TokenizerPayloadSerializer.DeserializeWordVocab(payload);
        return new WordTokenizer(vocab);
    }
}