import torch
from models import LightweightUNet, HeavyUNet
from thop import profile


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare():
    print("initializing models...")
    # 1. Load Models
    heavy_model = HeavyUNet()
    light_model = LightweightUNet()

    # 2. Count Parameters
    params_heavy = count_parameters(heavy_model)
    params_light = count_parameters(light_model)

    print(f"\n📉 FAIR ARCHITECTURE COMPARISON:")
    print(f"   Equivalent Standard U-Net: {params_heavy:,} parameters")
    print(f"   Your Lightweight U-Net:    {params_light:,} parameters")
    print(f"   Reduction Factor:          {params_heavy / params_light:.1f}x smaller")

    # 3. FLOPs
    input = torch.randn(1, 3, 256, 256)
    try:
        flops_heavy, _ = profile(heavy_model, inputs=(input,), verbose=False)
        flops_light, _ = profile(light_model, inputs=(input,), verbose=False)

        print(f"\n⚡ Speed/Cost Comparison (FLOPs):")
        print(f"   Equivalent Standard U-Net: {flops_heavy / 1e9:.2f} GFLOPs")
        print(f"   Your Lightweight U-Net:    {flops_light / 1e9:.2f} GFLOPs")
        print(f"   Efficiency Gain:           {flops_heavy / flops_light:.1f}x faster")
    except:
        pass


if __name__ == "__main__":
    compare()