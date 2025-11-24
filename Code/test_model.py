import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# --- Import your modules ---
from dataset import COCODataset
from models import Denoising_Model
from models import UNet
from utils import add_noise

# --- Config ---
DATA_PATH = '../data/imgs'
MODEL_PATH = 'denoising_ae_epoch_5.pth'  # <--- Verify this filename matches your saved file!
BATCH_SIZE = 8
IMG_SIZE = 128


def test():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧪 Testing on: {device}")

    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    # model = Denoising_Model().to(device)
    model = UNet().to(device)
    if torch.cuda.is_available():
        checkpoint = torch.load(MODEL_PATH)
    else:
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint)
    model.eval()  # Important: freezes BatchNorm/Dropout
    print("✅ Model loaded successfully.")

    # 3. Load Data
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    dataset = COCODataset(root_dir=DATA_PATH, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Get one batch
    clean_images = next(iter(loader)).to(device)

    # 4. Corrupt Data
    noisy_images = add_noise(clean_images, noise_type='gaussian', factor=0.5).to(device)

    # 5. Run Inference
    with torch.no_grad():
        reconstructed_images = model(noisy_images)

    # 6. Save Result Image
    print("Generating comparison image...")

    # Move to CPU for plotting
    clean = clean_images.cpu()
    noisy = noisy_images.cpu()
    recon = reconstructed_images.cpu()

    # Create a grid plot
    plt.figure(figsize=(15, 6))
    for i in range(BATCH_SIZE):
        # Top row: Clean
        plt.subplot(3, BATCH_SIZE, i + 1)
        plt.imshow(clean[i].permute(1, 2, 0))
        plt.axis('off')
        if i == 0: plt.title("Original")

        # Middle row: Noisy
        plt.subplot(3, BATCH_SIZE, i + 1 + BATCH_SIZE)
        plt.imshow(noisy[i].permute(1, 2, 0))
        plt.axis('off')
        if i == 0: plt.title("Noisy Input")

        # Bottom row: Reconstructed
        plt.subplot(3, BATCH_SIZE, i + 1 + 2 * BATCH_SIZE)
        plt.imshow(recon[i].permute(1, 2, 0))
        plt.axis('off')
        if i == 0: plt.title("AI Output")

    plt.tight_layout()
    plt.savefig('final_result.png')
    print("🎉 Done! Saved result to 'final_result.png'")


if __name__ == "__main__":
    test()