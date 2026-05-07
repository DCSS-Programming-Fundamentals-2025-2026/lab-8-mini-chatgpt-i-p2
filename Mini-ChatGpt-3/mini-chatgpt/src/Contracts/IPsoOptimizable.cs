using System.Collections.Generic;

namespace MiniChatGpt.Contracts
{

    public interface IPsoOptimizable
    {
        float[] GetFlatWeights();

        void SetFlatWeights(float[] flatWeights);

        float CalculateTotalLoss(List<int> tokens);

        IPsoOptimizable Clone();
    }
}