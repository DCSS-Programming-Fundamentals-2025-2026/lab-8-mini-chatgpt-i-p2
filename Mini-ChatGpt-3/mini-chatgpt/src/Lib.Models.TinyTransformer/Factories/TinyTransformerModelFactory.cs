using System.Text.Json;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Factories;

public class TinyTransformerModelFactory
{
    // public static TinyTransformerModel CreateManually(int vocabSize, int embeddingSize, int headCount, int contextSize,
    //     int seed, TinyTransformerWeights weights)
    // {
    //     TinyTransformerConfig config =
    //         new TinyTransformerConfig(vocabSize, embeddingSize, headCount, contextSize, seed, weights);
    //
    //     return new TinyTransformerModel(config);
    // }

    public static TinyTransformerModel CreateAuto(TinyTransformerConfig config)
    {
        return new TinyTransformerModel(config);
    }
    
    public static TinyTransformerModel FromPayload(JsonElement payload)
    {
        var options = new JsonSerializerOptions 
        { 
            PropertyNameCaseInsensitive = true 
        };
        
        var config = payload.Deserialize<TinyTransformerConfig>(options);

        if (config == null)
        {
            throw new InvalidOperationException("Failed to deserialize TinyTransformerConfig from payload");
        }
        
        return new TinyTransformerModel(config);
    }
}