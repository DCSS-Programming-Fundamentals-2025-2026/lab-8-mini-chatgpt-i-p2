using System.Text.Json;
using MiniChatGPT.Contracts;

namespace NGram
{
    public class TrigramModel : ILanguageModel
    {
        public string ModelKind
        {
            get { return "trigram"; }
        }

        public int VocabSize
        {
            get { return _bigramProbs.Length; }
        }

        public string GetContractFingerprint()
        {
            return "TrigramModel-1.0";
        }

        private Dictionary<(int, int), float[]> _trigramProbs { get; set; }
        private float[][] _bigramProbs { get; set; }
        private NGramCounts _counts;

        public TrigramModel(int vocabSize)
        {
            _trigramProbs = new Dictionary<(int, int), float[]>();
            _bigramProbs = new float[vocabSize][];

            for (int i = 0; i < vocabSize; i++)
            {
                _bigramProbs[i] = new float[vocabSize];
            }

            _counts = new NGramCounts(vocabSize);
        }

        public void Train(ReadOnlySpan<int> tokens)
        {
            _counts = new NGramCounts(_bigramProbs.Length);
            _trigramProbs.Clear();

            _counts.CountTrigrams(tokens);
            _counts.CountBigrams(tokens);

            for (int i = 0; i < _bigramProbs.Length; i++)
            {
                float rowSum = _counts.BigramCounts[i].Sum();

                if (rowSum > 0)
                {
                    for (int j = 0; j < _bigramProbs[i].Length; j++)
                    {
                        _bigramProbs[i][j] = _counts.BigramCounts[i][j] / rowSum;
                    }
                }
            }

            foreach (var item in _counts.TrigramCounts)
            {
                float rowSum = item.Value.Sum();

                if (rowSum > 0)
                {
                    if (!_trigramProbs.TryGetValue(item.Key, out float[] normalized))
                    {
                        normalized = new float[item.Value.Length];
                        _trigramProbs[item.Key] = normalized;
                    }

                    for (int i = 0; i < item.Value.Length; i++)
                    {
                        normalized[i] = item.Value[i] / rowSum;
                    }
                }
            }
        }

        public float[] NextTokenScores(ReadOnlySpan<int> context)
        {
            if (context.IsEmpty)
            {
                float[] uniform = new float[_bigramProbs.Length];
                for (int k = 0; k < uniform.Length; k++)
                {
                    uniform[k] = 1f / _bigramProbs.Length;
                }

                return uniform;
            }

            if (context.Length >= 2)
            {
                int p2 = context[context.Length - 2];
                int p1 = context[context.Length - 1];
                (int, int) key = (p2, p1);

                if (_trigramProbs.TryGetValue(key, out float[] trigramScores))
                {
                    float[] copy = new float[trigramScores.Length];
                    Array.Copy(trigramScores, copy, copy.Length);

                    return copy;
                }
            }

            int last = context[context.Length - 1];
            float[] fallbackCopy = new float[_bigramProbs[last].Length];
            Array.Copy(_bigramProbs[last], fallbackCopy, fallbackCopy.Length);

            return fallbackCopy;
        }

        public object GetPayloadForCheckpoint()
        {
            NGram.NGramPayloadMapper mapper = new NGram.NGramPayloadMapper();

            mapper.BigramProbs = _bigramProbs;

            System.Collections.Generic.List<NGram.TrigramEntry> trigramList = new System.Collections.Generic.List<NGram.TrigramEntry>();

            foreach (var kvp in _trigramProbs)
            {
                NGram.TrigramEntry entry = new NGram.TrigramEntry();

                entry.Prev2 = kvp.Key.Item1;
                entry.Prev1 = kvp.Key.Item2;
                entry.NextTokenScores = kvp.Value;

                trigramList.Add(entry);
            }

            mapper.TrigramProbs = trigramList;

            return mapper;
        }

        public void FromPayload(JsonElement payload)
        {
            NGramPayloadMapper data = payload.Deserialize<NGramPayloadMapper>();

            if (data.BigramProbs != null)
            {
                _bigramProbs = data.BigramProbs;
            }

            if (data.TrigramProbs != null)
            {
                _trigramProbs.Clear();

                foreach (var entry in data.TrigramProbs)
                {
                    (int, int) key = (entry.Prev2, entry.Prev1);
                    _trigramProbs[key] = entry.NextTokenScores;
                }
            }
        }
    }
}