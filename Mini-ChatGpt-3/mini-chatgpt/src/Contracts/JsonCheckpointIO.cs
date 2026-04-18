using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace MiniChatGPT.Contracts;

/// <summary>JSON implementation of ICheckpointIO for saving/loading checkpoints.</summary>
public sealed class JsonCheckpointIO : ICheckpointIO
{
    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNameCaseInsensitive = true,
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        IncludeFields = true
    };

    public void Save(string path, Checkpoint cp)
    {
        string rawJson = JsonSerializer.Serialize(cp, Options);

        var parsedNode = JsonNode.Parse(rawJson);

        var prettyOptions = new JsonSerializerOptions { WriteIndented = true };
        string prettyJson = parsedNode.ToJsonString(prettyOptions);

        File.WriteAllText(path, prettyJson);
    }

    public Checkpoint Load(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Файл чекпоінту не знайдено: {path}");
        }

        string json = File.ReadAllText(path);

        var dto = JsonSerializer.Deserialize<CheckpointDto>(json, Options)
                  ?? throw new InvalidOperationException("Файл чекпоінту порожній");

        return new Checkpoint(
            dto.ModelKind,
            dto.TokenizerKind,
            dto.TokenizerPayload.Clone(),
            dto.ModelPayload.Clone(),
            dto.Seed,
            dto.ContractFingerprintChain
        );
    }

    private sealed record CheckpointDto(
        string ModelKind,
        string TokenizerKind,
        JsonElement TokenizerPayload,
        JsonElement ModelPayload,
        int Seed,
        string ContractFingerprintChain
    );
}