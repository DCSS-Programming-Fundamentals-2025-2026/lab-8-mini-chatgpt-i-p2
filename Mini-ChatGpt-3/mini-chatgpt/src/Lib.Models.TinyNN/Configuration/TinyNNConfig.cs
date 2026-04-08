namespace Lib.Models.TinyNN.Configuration;

public record TinyNNConfig(
    int VocabSize,
    int EmbeddingSize = 32,
    int ContextSize = 8
);
