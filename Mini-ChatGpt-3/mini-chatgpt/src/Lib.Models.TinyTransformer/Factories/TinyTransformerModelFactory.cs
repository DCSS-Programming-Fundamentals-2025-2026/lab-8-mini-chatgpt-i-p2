using System.Text.Json;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Factories;

public static class TinyTransformerModelFactory
{
    public static TinyTransformerModel CreateAuto(TinyTransformerConfig config)
    {
        return new TinyTransformerModel(config);
    }

    public static TinyTransformerModel FromPayload(JsonElement payload, int vocabSize, Lib.MathCore.IMathOps mathOps)
    {
        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            IncludeFields = true
        };

        Lib.Models.TinyTransformer.State.TinyTransformerWeights weights =
            payload.Deserialize<Lib.Models.TinyTransformer.State.TinyTransformerWeights>(options);

        if (weights == null)
        {
            throw new Exception("Не вдалося завантажити ваги");
        }

        Lib.Models.TinyTransformer.Configuration.TinyTransformerConfig config =
            new Lib.Models.TinyTransformer.Configuration.TinyTransformerConfig(vocabSize, weights: weights);

        config.MathOps = mathOps;

        if (config.ContextSize == 0) config.ContextSize = 256;
        if (config.EmbeddingSize == 0) config.EmbeddingSize = 64; 
        if (config.HeadCount == 0) config.HeadCount = 4;

        if (weights.EmbeddingWeight != null)
        {
            config.TokenEmbeddings = weights.EmbeddingWeight;
        }

        return new TinyTransformerModel(config);
    }
}