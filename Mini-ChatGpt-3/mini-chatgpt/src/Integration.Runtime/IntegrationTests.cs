using MiniChatGPT.Contracts;
using MiniChatGPT.ChatConsole;
using MiniChatGPT.ChatConsole.Commands;
using MiniChatGPT.Sampling;
using Lib.MathCore; 
using Lib.Runtime;

namespace Integration.Runtime
{
    public class IntegrationTests
    {
        private class StubLanguageModel : ILanguageModel
        {
            public string ModelKind => "FakeIntegrationModel";
            public int VocabSize => 10;
            public string GetContractFingerprint() => "fake-fingerprint";
            public object GetPayloadForCheckpoint() => null;

            public float[] NextTokenScores(ReadOnlySpan<int> context)
            {
                return new float[] { 0.5f, 2.5f, 0.5f, 3.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f };
            }
        }

        private class StubTokenizer : MiniChatGPT.Contracts.ITokenizer
        {
            public int VocabSize => 10;
            public int[] Encode(string text) => new int[] { 1 };
            public string Decode(int[] tokens) => tokens.Length > 0 ? $"[Token_{tokens[0]}]" : "";
            public string Decode(ReadOnlySpan<int> tokens) => tokens.Length > 0 ? $"[Token_{tokens[0]}]" : "";
            public string GetContractFingerprint() => "stub-tokenizer";
            public object GetPayloadForCheckpoint() => null;
        }

        private class StubTextGenerator : ITextGenerator
        {
            public string Generate(string prompt, int maxTokens, float temperature, int topK, int? seed = null)
            {
                return "Fake response";
            }
        }

        private class BasicMathOps : IMathOps
        {
            public float[] LastReceivedProbs { get; private set; }

            public float[] Softmax(ReadOnlySpan<float> logits)
            {
                float max = logits.ToArray().Max();
                float sum = 0f;
                float[] result = new float[logits.Length];

                for (int i = 0; i < logits.Length; i++)
                {
                    result[i] = MathF.Exp(logits[i] - max);
                    sum += result[i];
                }

                for (int i = 0; i < logits.Length; i++)
                {
                    result[i] /= sum;
                }

                return result;
            }

            public int SampleFromProbs(ReadOnlySpan<float> probs, Random rng)
            {
                LastReceivedProbs = probs.ToArray();

                float p = (float)rng.NextDouble();
                float cumulative = 0.0f;

                for (int i = 0; i < probs.Length; i++)
                {
                    cumulative += probs[i];

                    if (p <= cumulative)
                    {
                        return i;
                    }
                }
                return probs.Length - 1;
            }

            public int ArgMax(ReadOnlySpan<float> scores)
            {
                int maxIndex = 0;
                for (int i = 1; i < scores.Length; i++)
                {
                    if (scores[i] > scores[maxIndex])
                    {
                        maxIndex = i;
                    }
                }
                return maxIndex;
            }

            public float CrossEntropyLoss(ReadOnlySpan<float> logits, int target)
            {
                return 0f;
            }
        }

        [Test]
        public void ChatRepl_CreatedWithStubGenerator()
        {
            var stubGenerator = new StubTextGenerator();

            Assert.DoesNotThrow(() =>
            {
                var repl = new ChatRepl(stubGenerator);
            });
        }

        [Test]
        public void ChatRepl_CommandsAreValidate()
        {
            var stubGenerator = new StubTextGenerator();
            var options = new ReplOptions { Temperature = 0.5f };
            var printedMessages = new List<string>();
            var context = new CommandExecutionContext(options, stubGenerator, msg => printedMessages.Add(msg));
            var registry = new CommandRegistry();

            bool isCommand = registry.TryExecute("/temp 3", context);

            Assert.That(isCommand, Is.True);
            Assert.That(options.Temperature, Is.EqualTo(3f));
            Assert.That(printedMessages.Any(m => m.Contains("оновлено") || m.Contains("Температуру")), Is.True);
        }

        [Test]
        public void Sampler_DeterminismWithSeed()
        {
            var mathOps = new BasicMathOps();
            var sampler = new Sampler(mathOps);
            float[] probs = { 0.1f, 0.4f, 0.3f, 0.2f };
            float temp = 1.0f;
            int topK = 3;
            int seed = 1337;

            int firstRunToken = sampler.SampleWithSeed(probs, temp, topK, seed);
            int secondRunToken = sampler.SampleWithSeed(probs, temp, topK, seed);

            Assert.That(firstRunToken, Is.EqualTo(secondRunToken));
        }

        [Test]
        public void Sampler_TemperatureInfluence_ChangesProbability()
        {
            var mathOps = new BasicMathOps();
            var sampler = new Sampler(mathOps);
            float[] probs = { 0.2f, 0.8f };
            int topK = 2;

            sampler.Sample(probs, 1.0f, topK);
            float normalTempProbability = mathOps.LastReceivedProbs[0];

            sampler.Sample(probs, 0.1f, topK); 
            float lowTempProbability = mathOps.LastReceivedProbs[0];

            Assert.That(lowTempProbability, Is.Not.EqualTo(normalTempProbability));
        }

        [Test]
        public void RuntimeTextGenerator_WithRealComponents()
        {
            var model = new StubLanguageModel();
            var mathOps = new BasicMathOps();
            var sampler = new Sampler(mathOps);
            var tokenizer = new StubTokenizer();

            var generator = new RuntimeTextGenerator(model, sampler, mathOps, tokenizer, false);

            string generatedText = generator.Generate(
                prompt: "тест",
                maxTokens: 3,
                temperature: 0.7f,
                topK: 5,
                seed: 42
            );

            Assert.That(generatedText, Is.Not.Null.And.Not.Empty);
            Assert.That(generatedText.Contains("[Token_"), Is.True);
        }

        [Test]
        public void RuntimeTextGenerator_StopsGeneration_OnEndMarker()
        {
            var model = new EndMarkerModel();
            var mathOps = new BasicMathOps();
            var sampler = new Sampler(mathOps);
            var tokenizer = new StubTokenizer();

            var generator = new RuntimeTextGenerator(model, sampler, mathOps, tokenizer, false);

            string generatedText = generator.Generate("старт", 100, 1.0f, 10, 100);
            var tokens = generatedText.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            Assert.That(tokens.Length, Is.LessThan(100));
        }

        private class EndMarkerModel : ILanguageModel
        {
            public string ModelKind => "EndMarkerModel";
            public int VocabSize => 10;
            public string GetContractFingerprint() => "";
            public object GetPayloadForCheckpoint() => null;

            private int _calls = 0;

            public float[] NextTokenScores(ReadOnlySpan<int> context)
            {
                float[] logits = new float[VocabSize];

                if (_calls == 0)
                {
                    logits[5] = 100.0f; 
                }
                else
                {
                    logits[0] = 100.0f; 
                }
                _calls++;

                return logits;
            }
        }
    }
}
