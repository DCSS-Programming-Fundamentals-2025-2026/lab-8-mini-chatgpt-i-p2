using System.Text.Json;
using CommandLine;
using Lib.Corpus.Configuration;
using Lib.Corpus.Domain;
using Lib.Corpus.Infrastructure;
using Lib.Corpus.Processing;
using Lib.MathCore;
using Lib.Models.TinyNN;
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.Factories;
using Lib.Models.TinyTransformer;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Factories;
using Lib.Models.TinyTransformer.Training;
using Lib.Tokenization;
using MiniChatGPT.Contracts;
using NGram;
using NGram.ModelFactory;

namespace Trainer
{

    public class Trainer
    {
        public static void Main(string[] args)
        {
            Parser.Default.ParseArguments<Options>(args).WithParsed(RunOptions).WithNotParsed(HandleParseError);
        }

        public static void RunOptions(Options opts)
        {
            Console.WriteLine($"Модель: {opts.Model}");

            string TokenizerVer = "Unknown";
            string CorpusVer = "Unknown";
            string ModelVer = "Unknown";

            var splitter = new CorpusSplitter();
            var normilizer = new CorpusTextNormalizer();
            var fileSystem = new DefaultFileSystem();
            var loader = new CorpusLoader(normilizer, splitter, fileSystem);

            CorpusVer = loader.GetContractFingerprint();

            string dataPath = opts.Data;

            if (!File.Exists(dataPath))
            {
                Console.WriteLine("Файл не знайдено");
                return;
            }

            CorpusLoadOptions loadOptions = new();
            var corpus = loader.Load(dataPath, loadOptions);

            Console.WriteLine($"Корпус довжиною {corpus?.TrainText.Length} завантажено");
            ITokenizerFactory tokenizerFactory = new WordTokenizerFactory();

            if (opts.Tokenizer == "word")
            {
                tokenizerFactory = new WordTokenizerFactory();
            }
            else if (opts.Tokenizer == "char")
            {
                tokenizerFactory = new CharTokenizerFactory();
            }
            else
            {
                Console.WriteLine("Невірний тип токенізатора");
                return;
            }

            Lib.Tokenization.ITokenizer tokenizer = tokenizerFactory.BuildFromText(corpus.TrainText);
            TokenizerVer = tokenizer.GetContractFingerprint();
            int[] codedTrainTokens = tokenizer.Encode(corpus.TrainText);

            Console.WriteLine($"Розмір словника: {tokenizer.VocabSize}");
            Console.WriteLine($"Всього токенів: {codedTrainTokens.Length}");

            IMathOps mathOps = new MathOpsImpl();
            JsonCheckpointIO json = new JsonCheckpointIO();

            Directory.CreateDirectory(Path.GetDirectoryName(opts.Out) ?? "data");

            NGramModelFactory modelFactory = new NGramModelFactory();

            if (opts.Model.ToLower() == "tinynn")
            {
                var nnFactory = new TinyNNModelFactory();
                var nnConfig = new TinyNNConfig(tokenizer.VocabSize, 64);
                TinyNNModel model;

                if (File.Exists(opts.Out))
                {
                    Checkpoint oldCheckpoint = json.Load(opts.Out);

                    if (oldCheckpoint.ModelKind.ToLower() == "tinynn")
                    {
                        JsonElement payload = (JsonElement)oldCheckpoint.ModelPayload;
                        model = (TinyNNModel)nnFactory.FromPayload(payload, tokenizer.VocabSize, mathOps);
                    }
                    else
                    {
                        model = nnFactory.CreateNew(nnConfig, mathOps);
                    }
                }
                else
                {
                    model = nnFactory.CreateNew(nnConfig, mathOps);
                }

                ModelVer = model.GetContractFingerprint();
                Console.WriteLine("Навчання TinyNN");

                Console.WriteLine($"Початок тренування у {opts.Epochs} епох");
                for (int i = 0; i < opts.Epochs; i++)
                {
                    float totalLoss = 0;
                    int contextSize = 8;
                    int count = 0;

                    for (int j = 0; j < codedTrainTokens.Length - contextSize; j++)
                    {
                        ReadOnlySpan<int> context = new ReadOnlySpan<int>(codedTrainTokens, j, contextSize);
                        int target = codedTrainTokens[j + contextSize];

                        float loss = model.TrainStep(context, target, opts.LearningRate);
                        totalLoss += loss;
                        count++;
                    }
                    Console.WriteLine($"Епоха {i + 1}/{opts.Epochs} - Втрата: {totalLoss / count:F4}");
                }

                Checkpoint checkpoint = new Checkpoint(opts.Model, opts.Tokenizer, tokenizer.GetPayloadForCheckpoint(), model.ToPayload(), opts.Seed, GenerateFingerprintChain(CorpusVer, TokenizerVer, ModelVer));
                json.Save(opts.Out, checkpoint);
            }
            else if (opts.Model.ToLower() == "tinytransformer")
            {
                TinyTransformerModel model;
                if (File.Exists(opts.Out))
                {
                    Checkpoint oldCheckpoint = json.Load(opts.Out);

                    if (oldCheckpoint.ModelKind.ToLower() == "tinytransformer")
                    {
                        JsonElement payload = (JsonElement)oldCheckpoint.ModelPayload;
                        model = TinyTransformerModelFactory.FromPayload(payload);
                    }
                    else
                    {
                        var tfConfig = new TinyTransformerConfig(tokenizer.VocabSize, 16, 1, 8, opts.Seed);
                        model = TinyTransformerModelFactory.CreateAuto(tfConfig);
                    }
                }
                else
                {
                    var tfConfig = new TinyTransformerConfig(tokenizer.VocabSize, 16, 1, 8, opts.Seed);
                    model = TinyTransformerModelFactory.CreateAuto(tfConfig);
                }
                ModelVer = model.GetContractFingerprint();
                Console.WriteLine("TinyTransformer створено");
                
                for (int i = 0; i < opts.Epochs; i++)
                {
                    for (int j = 0; j < codedTrainTokens.Length - model._config.ContextSize; j++)
                    {
                        ReadOnlySpan<int> context = new ReadOnlySpan<int>(codedTrainTokens, j, model._config.ContextSize);
                        int target = codedTrainTokens[j + model._config.ContextSize];

                        Training.Train(model, context, target, opts.LearningRate);
                    }
                }
                
                Checkpoint checkpoint = new Checkpoint(opts.Model, opts.Tokenizer, tokenizer.GetPayloadForCheckpoint(), model.GetPayloadForCheckpoint(), opts.Seed, GenerateFingerprintChain(CorpusVer, TokenizerVer, ModelVer));
                json.Save(opts.Out, checkpoint);
            }
            else if (opts.Model.ToLower() == "trigram")
            {
                TrigramModel trigram = modelFactory.CreateTrigramModel((tokenizer.VocabSize));
                ModelVer = trigram.GetContractFingerprint();
                Console.WriteLine("Trigram створено");

                trigram.Train(codedTrainTokens);

                Checkpoint checkpoint = new Checkpoint(opts.Model, opts.Tokenizer, tokenizer.GetPayloadForCheckpoint(), trigram.GetPayloadForCheckpoint(), opts.Seed, GenerateFingerprintChain(CorpusVer, TokenizerVer, ModelVer));
                json.Save(opts.Out, checkpoint);
            }
            else if (opts.Model.ToLower() == "bigram")
            {
                NGramModel bigram = modelFactory.CreateBigramModel((tokenizer.VocabSize));
                ModelVer = bigram.GetContractFingerprint();
                Console.WriteLine("Bigram створено");

                bigram.Train(codedTrainTokens);

                Checkpoint checkpoint = new Checkpoint(opts.Model, opts.Tokenizer, tokenizer.GetPayloadForCheckpoint(), bigram.GetPayloadForCheckpoint(), opts.Seed, GenerateFingerprintChain(CorpusVer, TokenizerVer, ModelVer));
                json.Save(opts.Out, checkpoint);
            }
            else
            {
                Console.WriteLine("Невірна модель");
            }

            Console.WriteLine($"Тренування завершилося, прогрес збережено у {opts.Out}");
        }

        public static void HandleParseError(IEnumerable<Error> errs)
        {
            Console.WriteLine("Неправильні параметри");
        }

        private static string GenerateFingerprintChain(string corpusLoaderVer, string tokenizerVer, string modelVer)
        {
            return $"Lib.Corpus: {corpusLoaderVer}|Lib.Tokenization: {tokenizerVer}|Lib.Model: {modelVer}";
        }
    }
}