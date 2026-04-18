using Lib.MathCore;
using MiniChatGPT.Contracts;
using MiniChatGPT.Sampling.Interfaces;

namespace Lib.Runtime
{
    public class RuntimeTextGenerator : ITextGenerator
    {
        private readonly ILanguageModel _model;
        private readonly ISampler _sampler;
        private readonly IMathOps _mathOps;
        private readonly ITokenizer _tokenizer;
        private readonly bool _requiresSoftmax;

        public RuntimeTextGenerator(
            ILanguageModel model,
            ISampler sampler,
            IMathOps mathOps,
            ITokenizer tokenizer,
            bool requiresSoftmax)
        {
            this._model = model ?? throw new ArgumentNullException(nameof(model));
            this._sampler = sampler ?? throw new ArgumentNullException(nameof(sampler));
            this._mathOps = mathOps ?? throw new ArgumentNullException(nameof(mathOps));
            this._tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            this._requiresSoftmax = requiresSoftmax;
        }

        public string Generate(string prompt, int maxTokens, float temperature, int topK, int? seed = null)
        {
            if (string.IsNullOrEmpty(prompt))
            {
                return string.Empty;
            }

            int[] initialTokens = this._tokenizer.Encode(prompt);
            List<int> currentContext = new List<int>(initialTokens);
            List<int> generatedTokens = new List<int>();

            Random rng = seed.HasValue ? new Random(seed.Value) : new Random();

            for (int i = 0; i < maxTokens; i++)
            {
                float[] scores = this._model.NextTokenScores(currentContext.ToArray());

                float[] probabilities;

                if (this._requiresSoftmax)
                {
                    probabilities = this._mathOps.Softmax(scores);
                }
                else
                {
                    probabilities = scores;
                }

                int nextTokenId = this._sampler.Sample(probabilities, temperature, topK, rng);

                currentContext.Add(nextTokenId);
                generatedTokens.Add(nextTokenId);

                string decodedToken = this._tokenizer.Decode(new int[] { nextTokenId });

                if (generatedTokens.Count >= 1)
                {
                    if (decodedToken.Contains("!") || decodedToken.Contains("?"))
                    {
                        break;
                    }
                }
            }

            return this._tokenizer.Decode(generatedTokens.ToArray());
        }
    }
}