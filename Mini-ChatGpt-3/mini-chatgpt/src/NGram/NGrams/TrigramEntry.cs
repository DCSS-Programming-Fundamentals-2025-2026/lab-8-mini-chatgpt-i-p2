namespace NGram
{
    public class TrigramEntry
    {
        public int Prev2
        {
            get;
            set;
        }

        public int Prev1
        {
            get;
            set;
        }

        public float[] NextTokenScores
        {
            get;
            set;
        }
    }
}
