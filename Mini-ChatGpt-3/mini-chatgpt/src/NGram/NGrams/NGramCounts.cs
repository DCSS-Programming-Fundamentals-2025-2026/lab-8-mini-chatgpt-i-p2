using System.Diagnostics.Tracing;

public class NGramCounts
{
    public float[][] BigramCounts { get; set; }
    public Dictionary<(int, int), float[]> TrigramCounts { get; set; }

    public int VocabSize { get; set; }

    public NGramCounts(int vocabSize)
    {
        VocabSize = vocabSize;

        BigramCounts = new float[vocabSize][];

        TrigramCounts = new Dictionary<(int, int), float[]>();

        for(int i = 0; i < vocabSize; i++)
        {
            BigramCounts[i] = new float[vocabSize];
        }
    }

    public void CountBigrams(ReadOnlySpan<int> tokens)
    {
        for(int i = 0; i < tokens.Length - 1; i++)
        {
            BigramCounts[tokens[i]][tokens[i+1]]++;
        }
    }

    public void CountTrigrams(ReadOnlySpan<int> tokens)
    {
        for (int i = 0; i < tokens.Length - 2; i++)
        {
            (int, int) key = (tokens[i], tokens[i + 1]);
            bool isContain = TrigramCounts.ContainsKey(key);
            if (isContain)
            {
                float[] array = TrigramCounts[key];
                array[tokens[i + 2]]++;
            }
            else
            {
                float[] newArray = new float[VocabSize];
                newArray[tokens[i + 2]]++;
                TrigramCounts.Add(key, newArray);
            }
        }
    }
}