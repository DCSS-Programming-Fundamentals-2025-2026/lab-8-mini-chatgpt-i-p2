namespace Lib.Models.TinyTransformer;

public class MatrixHelper
{
    public static float[][] MultiplyMatrix(float[][] matrix, float[][] weights)
    {

        if (matrix[0].Length != weights.Length)
        {
            throw new ArgumentException("Matrix dimensions do not match");
        }
        
        float[][] res = new float[matrix.Length][];

        for (int i = 0; i < res.Length; i++)
        {
            res[i] = new float[weights[0].Length];

            for (int j = 0; j < res[i].Length; j++)
            {
                float sum = 0;

                for (int k = 0; k < matrix.Length; k++)
                {
                    sum += matrix[i][k] * weights[k][j];
                }
                
                res[i][j] = sum;
            }
        }

        return res;
    }
    
    public static float[][] TransposeMatrix(float[][] matrix)
    {
        float[][] res = new float[matrix[0].Length][];

        for (int i = 0; i < res.Length; i++)
        {
            res[i] = new float[matrix.Length];

            for (int j = 0; j < res[i].Length; j++)
            {
                res[i][j] = matrix[j][i];
            }
        }

        return res;
    }

    public static void LineSumm(float[] line1 , float[] line2)
    {
        if (line1.Length != line2.Length)
        {
            throw new IndexOutOfRangeException();
        }

        for (int i = 0; i < line1.Length; i++)
        {
            line1[i] += line2[i];
        }
    }
}