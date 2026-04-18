using Lib.MathCore;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Layers;

namespace Lib.Models.TinyTransformer
{
    public class TinyTransformerModel
    {
        private readonly TinyTransformerConfig _config;
        private readonly MathOpsImpl _mathOps = new MathOpsImpl();
        private readonly string Version = "1.0.0";

        public TinyTransformerModel(TinyTransformerConfig config)
        {
            this._config = config;
        }

        public string GetContractFingerprint()
        {
            return this.Version;
        }

        public float[] NextTokenScores(int[] context)
        {
            TransformerBlock block = new TransformerBlock(this._config);

            float[] logits = block.Forward(context);

            return logits;
        }

        public object GetPayloadForCheckpoint()
        {
            return this._config;
        }
    }
}