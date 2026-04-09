using System;
using MiniChatGPT.Contracts;
using Lib.Models.TinyNN;
using Lib.Models.TinyTransformer;

namespace MiniChatGPT.Chat
{
    public class LanguageModelAdapter : ILanguageModel
    {
        private readonly object _originalModel;
        private readonly string _modelKind;
        private readonly int _vocabSize;

        public LanguageModelAdapter(object originalModel, string modelKind, int vocabSize)
        {
            _originalModel = originalModel;
            _modelKind = modelKind;
            _vocabSize = vocabSize;
        }

        public string ModelKind
        {
            get
            {
                return _modelKind;
            }
        }

        public int VocabSize
        {
            get
            {
                return _vocabSize;
            }
        }

        public float[] NextTokenScores(ReadOnlySpan<int> context)
        {
            if (_modelKind == "tinynn")
            {
                TinyNNModel nn = (TinyNNModel)_originalModel;

                return nn.NextTokenScores(context);
            }
            else
            {
                if (_modelKind == "tinytransformer")
                {
                    TinyTransformerModel tf = (TinyTransformerModel)_originalModel;

                    return tf.NextTokenScores(context.ToArray());
                }
                else
                {
                    throw new InvalidOperationException("Невідомий тип моделі");
                }
            }
        }

        public object GetPayloadForCheckpoint()
        {
            if (_modelKind == "tinynn")
            {
                TinyNNModel nn = (TinyNNModel)_originalModel;

                return nn.ToPayload();
            }
            else
            {
                if (_modelKind == "tinytransformer")
                {
                    TinyTransformerModel tf = (TinyTransformerModel)_originalModel;

                    return tf.GetPayloadForCheckpoint();
                }
                else
                {
                    return null;
                }
            }
        }

        public string GetContractFingerprint()
        {
            return "LanguageModelAdapter-Integration";
        }
    }
}