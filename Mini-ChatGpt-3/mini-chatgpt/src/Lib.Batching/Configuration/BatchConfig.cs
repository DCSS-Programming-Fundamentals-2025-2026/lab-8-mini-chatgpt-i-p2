public class BatchConfig
{
    public int BatchSize { get; }
    public int BlockSize { get; }
    public BatchConfig(int batchSize, int blockSize)
    {
        BatchSize = batchSize;
        BlockSize = blockSize;
    }
}