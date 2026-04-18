namespace Lib.Batching
{
    public interface IBatchProvider
    {
        Batch GetBatch(int batchSize, int blockSize, Random? rng);
    }
}