using System.Text.Json;

namespace Lib.Tokenization;

public interface ITokenizerFactory
{
    ITokenizer BuildFromText(string text);
    ITokenizer FromPayload(JsonElement payload);
}
