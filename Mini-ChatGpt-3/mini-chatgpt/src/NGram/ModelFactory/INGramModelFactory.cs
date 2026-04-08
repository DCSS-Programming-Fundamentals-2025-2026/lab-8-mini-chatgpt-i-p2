using System.Text.Json;

namespace NGram.ModelFactory
{
    public interface INGramModelFactory
    {
        public TrigramModel CreateTrigramModel(int vocabSize);
        public NGramModel CreateBigramModel(int vocabSize);

        TrigramModel CreateTrigramModelFromPayload(int vocabSize, JsonElement payload);
        NGramModel CreateBigramModelFromPayload(int vocabSize, JsonElement payload);
    }
}
