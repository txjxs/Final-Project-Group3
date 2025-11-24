import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os

# --- Import your modules ---
from dataset import COCODataset
from models import Denoising_Model, UNet
from utils import add_noise

# --- Configuration ---
DATA_PATH = '../data/imgs'  # Check this path matches your folder structure!
BATCH_SIZE = 2048  # Reduce to 16 if you run out of memory
LEARNING_RATE = 0.001
EPOCHS = 5
NOISE_FACTOR = 0.4  # How much noise to destroy the image with
NUM_WORKERS = 8


def train():
    # 1. Setup Device (Use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # 2. Prepare Data
    print("Loading Data...")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = COCODataset(root_dir=DATA_PATH, transform=transform)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2,      # Force CPU to work 2 batches ahead
    persistent_workers=True)
    print(f"✅ Found {len(dataset)} images.")

    # 3. Initialize Model, Loss, and Optimizer
    # model = Denoising_Model().to(device)
    model = UNet().to(device)
    criterion = nn.MSELoss()  # Mean Squared Error (Standard for images)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print("🔥 Starting Training Loop...")

    for epoch in range(EPOCHS):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for batch_idx, clean_images in enumerate(train_loader):
            # A. Move data to device (GPU/CPU)
            clean_images = clean_images.to(device)

            # B. Create Noisy Input (The "Problem")
            # We do this on the fly so the model sees different noise every time
            noisy_images = add_noise(clean_images, noise_type='gaussian', factor=NOISE_FACTOR)
            noisy_images = noisy_images.to(device)

            # C. Zero Gradients
            optimizer.zero_grad()

            # D. Forward Pass (The Model guesses)
            outputs = model(noisy_images)

            # E. Compute Loss
            # CRITICAL: Compare Output vs CLEAN images (not noisy ones)
            loss = criterion(outputs, clean_images)

            # F. Backward Pass (Calculate adjustments)
            loss.backward()

            # G. Optimize (Update weights)
            optimizer.step()

            running_loss += loss.item()

            # Logging every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # End of Epoch Summary
        epoch_loss = running_loss / len(train_loader)
        print(f"-------- Epoch [{epoch + 1}/{EPOCHS}] Average Loss: {epoch_loss:.4f} --------")

        # Save Model Checkpoint
        torch.save(model.state_dict(), f"denoising_ae_epoch_{epoch + 1}.pth")
        print(f"💾 Model saved: denoising_ae_epoch_{epoch + 1}.pth")

    print("🎉 Training Complete!")


if __name__ == "__main__":
    train()