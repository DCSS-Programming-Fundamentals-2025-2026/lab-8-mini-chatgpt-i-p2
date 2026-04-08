namespace MiniChatGPT.Sampling.Processing
{
    public static class TopKSelector
    {
        public static int[] Select(float[] tempered, int topK)
        {
            int curK = Math.Min(topK, tempered.Length);

            int[] idx = new int[tempered.Length];

            for (int i = 0; i < idx.Length; i++)
            {
                idx[i] = i;
            }

            Array.Sort(idx, (a, b) =>
            {
                return tempered[b].CompareTo(tempered[a]);
            });

            int[] topKIdx = new int[curK];

            for(int i = 0; i < curK; i++)
            {
                topKIdx[i] = idx[i];
            }

            return topKIdx;
        }
    }
}
