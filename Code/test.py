import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import random  # Needed for random noise selection

# --- Modules ---
from dataset import COCODataset
# from models import LightweightUNet as UNet
# from models import HeavyUNet as UNet # Ensure we use the lightweight class
from models import Denoising_Model as UNet
from utils import add_noise, seed_everything, plot_denoising_result, calculate_psnr

# --- Config ---
SEED = 42
DATA_PATH = '../data/imgs'
MODEL_PATH = f'{UNet.name()}.pth'
# MODEL_PATH = 'Models/HeavyUNet.pth' # Matches train.py save name
BATCH_SIZE = 32
NOISE_FACTOR = 0.5


def test():
    # 1. Reproduce Setup
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧪 Testing on device: {device}")

    # 2. Recreate Splits
    transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(256),  # Center crop for consistent testing
        transforms.ToTensor(),
    ])

    # Handle Dataset Class Name
    full_dataset = COCODataset(root_dir=DATA_PATH, transform=transform) # Now explicitly using COCODataset

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    _, _, test_set = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    print(f"✅ Test Set Loaded: {len(test_set)} images")

    # 3. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found.")
        return

    model = UNet().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Model weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    model.eval()

    # 4. Evaluate
    total_psnr = 0.0

    print("Running Evaluation...")

    # Containers for the first batch to plot later
    sample_clean, sample_noisy, sample_recon = None, None, None
    sample_noise_type = ""

    with torch.no_grad():
        for i, clean_images in enumerate(test_loader):
            clean_images = clean_images.to(device)

            # --- UPDATE: Random Noise Selection (Matches Training) ---
            current_noise_type = random.choice(['gaussian', 'salt_pepper'])

            noisy_images = add_noise(clean_images, noise_type=current_noise_type, factor=NOISE_FACTOR).to(device)

            outputs = model(noisy_images)

            # Metrics
            psnr = calculate_psnr(outputs, clean_images)
            total_psnr += psnr.item()

            # Save the first batch for visualization
            if i == 0:
                sample_clean = clean_images
                sample_noisy = noisy_images
                sample_recon = outputs
                sample_noise_type = current_noise_type

    avg_psnr = total_psnr / len(test_loader)

    print(f"\n🏆 Final Test Results:")
    print(f"   Avg PSNR: {avg_psnr:.2f} dB")

    # 5. Save Visual Report
    plot_denoising_result(
        sample_clean,
        sample_noisy,
        sample_recon,
        psnr=avg_psnr,
        save_path='test_report.png',
        Model_name=f"Heavy U-Net ({sample_noise_type} noise)"
    )


if __name__ == "__main__":
    test()