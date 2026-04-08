namespace Lib.MathCore;


public static class MathOps
{

    private static readonly IMathOps _default = new MathOpsImpl();

    public static IMathOps Default => _default;
}