using MiniChatGPT.Contracts;
using Lib.Models.TinyNN;
using Lib.Models.TinyTransformer;
using Lib.Models.TinyTransformer.Training;

namespace MiniChatGPT.Chat
{
    public class LanguageModelAdapter : ILanguageModel
    {
        private readonly object _originalModel;
        private readonly string _modelKind;
        private readonly int _vocabSize;

        public LanguageModelAdapter(object originalModel, string modelKind, int vocabSize)
        {
            this._originalModel = originalModel;
            this._modelKind = modelKind;
            this._vocabSize = vocabSize;
        }

        public int VocabSize => this._vocabSize;
        public string ModelKind => this._modelKind;

        public bool RequiresSoftmax => (this._modelKind == "tinynn" || this._modelKind == "tinytransformer");

        public float[] NextTokenScores(ReadOnlySpan<int> context)
        {
            if (this._modelKind == "trigram" || this._modelKind == "bigram")
            {
                var m = (ILanguageModel)this._originalModel;
                return m.NextTokenScores(context);
            }

            if (this._modelKind == "tinynn")
            {
                return ((TinyNNModel)this._originalModel).NextTokenScores(context);
            }

            if (this._modelKind == "tinytransformer")
            {
                return ((TinyTransformerModel)this._originalModel).NextTokenScores(context.ToArray());
            }

            throw new Exception("Unknown model kind");
        }

        public object GetPayloadForCheckpoint()
        {
            if (this._modelKind == "tinynn")
            {
                TinyNNModel nn = (TinyNNModel)this._originalModel;
                return nn.ToPayload();
            }
            else if (this._modelKind == "tinytransformer")
            {
                TinyTransformerModel tf = (TinyTransformerModel)this._originalModel;
                return tf.GetPayloadForCheckpoint();
            }
            else if (this._modelKind == "trigram" || this._modelKind == "bigram")
            {
                return ((ILanguageModel)this._originalModel).GetPayloadForCheckpoint();
            }
            else
            {
                return null;
            }
        }

        public string GetContractFingerprint()
        {
            return "LanguageModelAdapter-1.0.0";
        }
    }
}