using System.Text.Json;

namespace NGram.ModelFactory
{
    public class NGramModelFactory : INGramModelFactory
    {
        public NGramModel CreateBigramModel(int vocabSize)
        {
            return new NGramModel(vocabSize);
        }

        public TrigramModel CreateTrigramModel(int vocabSize)
        {
            return new TrigramModel(vocabSize);
        }

        public NGramModel CreateBigramModelFromPayload(int vocabSize, JsonElement payload)
        {
            NGramModel model = new NGramModel(vocabSize);
            model.FromPayload(payload);

            return model;
        }

        public TrigramModel CreateTrigramModelFromPayload(int vocabSize, JsonElement payload)
        {
            TrigramModel model = new TrigramModel(vocabSize);
            model.FromPayload(payload);

            return model;
        }
    }
}
