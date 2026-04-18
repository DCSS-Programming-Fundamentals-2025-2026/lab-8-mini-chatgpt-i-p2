using System.Text.Json;
using Lib.Models.TinyNN.Configuration;
using Lib.MathCore;
using Lib.Models.TinyNN.State;

namespace Lib.Models.TinyNN.Factories;

public class TinyNNModelFactory
{
    public TinyNNModel CreateNew(TinyNNConfig config, IMathOps mathOps)
    {
        TinyNNWeights weights = new TinyNNWeights(config);
        return new TinyNNModel(config, weights, mathOps);
    }

    public TinyNNModel FromPayload(JsonElement payload, int vocabSize, IMathOps mathOps)
    {
        JsonElement configParams = payload.TryGetProperty("config", out var c) ? c : payload.GetProperty("Config");

        int embeddingSize = configParams.TryGetProperty("embeddingSize", out var e) ? e.GetInt32() : configParams.GetProperty("EmbeddingSize").GetInt32();
        int contextSize = configParams.TryGetProperty("contextSize", out var ctx) ? ctx.GetInt32() : configParams.GetProperty("ContextSize").GetInt32();

        var config = new TinyNNConfig(vocabSize, embeddingSize, contextSize);

        JsonElement weightParams = payload.TryGetProperty("weights", out var w) ? w : payload.GetProperty("Weights");

        var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var weights = JsonSerializer.Deserialize<TinyNNWeights>(weightParams.GetRawText(), options);

        if (weights == null)
        {
            throw new System.Exception("Ваги не завантажилися");
        }

        return new TinyNNModel(config, weights, mathOps);
    }
}