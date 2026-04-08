namespace MiniChatGPT.Sampling.Options
{
    public class SamplingOptions
    {
        public float Temperature { get; set; }
        public int TopK {  get; set; }
        public int? Seed {  get; set; }

        public SamplingOptions()
        {
            Temperature = 1.0f;
            TopK = 10;
            Seed = null;
        }
    }
}
