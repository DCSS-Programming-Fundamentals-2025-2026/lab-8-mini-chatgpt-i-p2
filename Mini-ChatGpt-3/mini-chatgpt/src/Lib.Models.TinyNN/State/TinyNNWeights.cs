using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;
using Lib.Models.TinyNN.Configuration;

namespace Lib.Models.TinyNN.State;
public class TinyNNWeights
{
    [JsonPropertyName("embeddings")]
    public float[][] Embeddings { get; set; }
    
    [JsonPropertyName("outputWeights")]
    public float[][] OutputWeights { get; set; }
    
    [JsonPropertyName("outputBias")]
    public float[] OutputBias { get; set; }

    private readonly float minRange = -0.1f;
    private readonly float maxRange = 0.1f;

    [JsonConstructor]
    public TinyNNWeights() { }
    
    public TinyNNWeights(TinyNNConfig config)
    {
        Random temp = new Random();
        Embeddings = new float[config.VocabSize][];
        for (int i = 0; i < config.VocabSize; i++)
        {
            Embeddings[i] = new float[config.EmbeddingSize];
            for (int j = 0; j < config.EmbeddingSize; j++)
            {
                Embeddings[i][j] = (float)temp.NextDouble() * (maxRange - minRange) + minRange;
            }
        }
        
        OutputWeights = new float[config.EmbeddingSize][];
        for (int i = 0; i < config.EmbeddingSize; i++)
        {
            OutputWeights[i] = new float[config.VocabSize];
            for (int j = 0; j < config.VocabSize; j++)
            {
                OutputWeights[i][j] = (float)temp.NextDouble() * (maxRange - minRange) + minRange;
            }
        }
        
        OutputBias = new float[config.VocabSize];
        for (int i = 0; i < config.VocabSize; i++)
        {
            OutputBias[i] = (float)temp.NextDouble() * (maxRange - minRange) + minRange;
        }
    }
}