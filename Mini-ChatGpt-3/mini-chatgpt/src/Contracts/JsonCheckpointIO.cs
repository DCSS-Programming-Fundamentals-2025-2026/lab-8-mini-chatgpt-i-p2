using System.Text.Json;

namespace MiniChatGPT.Contracts;

/// <summary>JSON implementation of ICheckpointIO for saving/loading checkpoints.</summary>
public sealed class JsonCheckpointIO : ICheckpointIO
{
    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = false,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    public void Save(string path, Checkpoint cp)
    {
        JsonSerializerOptions options = new JsonSerializerOptions()
        {
            WriteIndented = true
        };

        string jsonString = JsonSerializer.Serialize(cp, options);

        File.WriteAllText(path, jsonString);
    }

    public Checkpoint Load(string path)
    {
        string json = File.ReadAllText(path);

        var options = new System.Text.Json.JsonSerializerOptions()
        {
            PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase,
            PropertyNameCaseInsensitive = true
        };

        var dto = System.Text.Json.JsonSerializer.Deserialize<CheckpointDto>(json, options) ?? throw new InvalidOperationException("Файл чекпоінту порожній");

        return new Checkpoint(
            dto.ModelKind,               
            dto.ContractFingerprintChain, 
            dto.ModelPayload.Clone(),     
            dto.TokenizerPayload.Clone(), 
            dto.Seed,                     
            dto.TokenizerKind           
        );
    }

    private sealed record CheckpointDto(
    string ModelKind,
    string TokenizerKind,
    System.Text.Json.JsonElement TokenizerPayload,
    System.Text.Json.JsonElement ModelPayload,
    int Seed,
    string ContractFingerprintChain
    );
}
