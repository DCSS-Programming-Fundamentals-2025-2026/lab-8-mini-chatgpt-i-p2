namespace Lib.MathCore.Utilities;

public static class ScoreUtilities
{
    public static int GetArgMax(ReadOnlySpan<float> scores)
    {

        if (scores.Length == 0) 
        {
            return -1;
        }
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
}