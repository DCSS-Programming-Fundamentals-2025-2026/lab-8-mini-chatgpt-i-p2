namespace MiniChatGPT.Sampling.Processing
{
    public static class TemperatureScaler
    {
        public static float[] Scale(float[] probs, float temperature)
        {
            float[] tempered = new float[probs.Length];

            for (int i = 0; i < probs.Length; i++)
            {
                tempered[i] = MathF.Log(probs[i]) / temperature;
            }

            return tempered;
        }
    }
}
