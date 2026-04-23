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
    
    public static TinyTransformerModel FromPayload(JsonElement payload)
    {
        var options = new JsonSerializerOptions 
        {
            WriteIndented = true,
        };
        
        TinyTransformerConfig config = payload.Deserialize<TinyTransformerConfig>(options);

        if (config == null)
        {
            config = new TinyTransformerConfig(5000);
        }
        
        return new TinyTransformerModel(config);
    }
}