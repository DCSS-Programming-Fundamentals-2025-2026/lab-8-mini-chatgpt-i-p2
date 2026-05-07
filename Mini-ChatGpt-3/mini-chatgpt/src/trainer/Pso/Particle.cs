namespace Trainer.Pso;

public class Particle
{
    public float[] Position { get; set; }
    public float[] Velocity { get; set; }
    public float[] BestPosition { get; set; }
    public float BestLoss { get; set; } = float.MaxValue;

    public Particle(int paramCount, Random rnd)
    {
        Position = new float[paramCount];
        Velocity = new float[paramCount];
        BestPosition = new float[paramCount];

        for (int i = 0; i < paramCount; i++)
        {
            Position[i] = (float)(rnd.NextDouble() * 2 - 1);
            Velocity[i] = (float)(rnd.NextDouble() * 0.1);
        }
    }
}