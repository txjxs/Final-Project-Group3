import torch
from models import Denoising_Model

# 1. Create the model
model = Denoising_Model()

# 2. Create a fake image batch (Batch=4, Channels=3, Height=128, Width=128)
dummy_input = torch.randn(4, 3, 128, 128)
print(dummy_input)
# 3. Pass it through
output = model(dummy_input)

# 4. Check
print(f"Input Shape:  {dummy_input.shape}")
print(f"Output Shape: {output.shape}")

if dummy_input.shape == output.shape:
    print(" SUCCESS: Output shape matches Input shape!")
else:
    print("ERROR: Shapes do not match.")