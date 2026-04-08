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
        private readonly MiniChatGPT.Contracts.ITokenizer _tokenizer; 
        private readonly bool _requiresSoftmax;

        public RuntimeTextGenerator(ILanguageModel model, ISampler sampler, IMathOps mathOps, MiniChatGPT.Contracts.ITokenizer tokenizer, bool requiresSoftmax)
        {
            if (model == null)
            {
                throw new ArgumentNullException(nameof(model));
            }

            if (sampler == null)
            {
                throw new ArgumentNullException(nameof(sampler));
            }

            if (mathOps == null)
            {
                throw new ArgumentNullException(nameof(mathOps));
            }

            if (tokenizer == null)
            {
                throw new ArgumentNullException(nameof(tokenizer));
            }

            _model = model;
            _sampler = sampler;
            _mathOps = mathOps;
            _tokenizer = tokenizer;
            _requiresSoftmax = requiresSoftmax;
        }

        public string Generate(string prompt, int maxTokens, float temperature, int topK, int? seed = null)
        {
            if (string.IsNullOrEmpty(prompt))
            {
                prompt = " ";
            }

            int[] initialTokens = _tokenizer.Encode(prompt);
            List<int> currentContext = new List<int>(initialTokens);
            List<int> generatedTokens = new List<int>();

            Random rng = null;

            if (seed.HasValue)
            {
                rng = new Random(seed.Value);
            }

            for (int i = 0; i < maxTokens; i++)
            {
                int[] contextArray = currentContext.ToArray();
                float[] scores = _model.NextTokenScores(contextArray);

                float[] probabilities;

                if (_requiresSoftmax)
                {
                    probabilities = _mathOps.Softmax(scores);
                }
                else
                {
                    probabilities = scores;
                }

                int nextTokenId = _sampler.Sample(probabilities, temperature, topK, rng);

                currentContext.Add(nextTokenId);
                generatedTokens.Add(nextTokenId);

                int[] currentTokenArray = new int[1] { nextTokenId };
                string decodedToken = _tokenizer.Decode(currentTokenArray);

                if (generatedTokens.Count >= 5)
                {
                    if (decodedToken.Contains("!") || decodedToken.Contains("?"))
                    {
                        break;
                    }
                }
            }

            int[] finalGeneratedTokens = generatedTokens.ToArray();

            return _tokenizer.Decode(finalGeneratedTokens);
        }
    }
}