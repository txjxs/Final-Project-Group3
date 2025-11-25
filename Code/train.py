import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import multiprocessing

# --- Import your modules ---
from dataset import COCODataset
from models import Denoising_Model, UNet
from utils import add_noise, seed_everything

# --- Configuration ---
SEED = 42
DATA_PATH = '../data/imgs'
BATCH_SIZE = 500
LEARNING_RATE = 0.001
EPOCHS = 5
NOISE_FACTOR = 0.8
NUM_WORKERS = multiprocessing.cpu_count()


def train():
    seed_everything(SEED)
    # 1. Setup Device (Use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Prepare Data
    transform = transforms.Compose([
        transforms.Resize(320),  # Resize smallest side to 320 (keep aspect ratio)
        transforms.RandomCrop(256),  # Cut a random 256x256 patch
        transforms.RandomHorizontalFlip(),  # Extra free data augmentation
        transforms.ToTensor(),
    ])

    full_dataset = COCODataset(root_dir=DATA_PATH, transform=transform)
    total_size = len(full_dataset)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    print(f" Dataset Size: {total_size}")
    print(f"   Split Config: Train={train_size}, Val={val_size}, Test={test_size}")
    print(f"   Sum Check: {train_size + val_size + test_size} == {total_size}")

    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size],
                                                generator=torch.Generator().manual_seed(SEED))

    print(f'Training set size: {len(train_set)}'
          f'Validation set size: {len(val_set)}'
          f'Test set size: {len(test_set)}')

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
                            prefetch_factor=2,
                            persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, )

    # 3. Initialize Model, Loss, and Optimizer
    # model = Denoising_Model().to(device)
    model = UNet().to(device)
    criterion = nn.MSELoss()  # Mean Squared Error (Standard for images)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print("Starting Training Loop...")
    total_batch = len(train_loader)

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        ###TRAIN
        model.train()  # Set model to training mode
        train_loss = 0.0
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
            train_loss += loss.item()
            # Logging every 10 batches
            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}] Batch [{batch_idx}/{total_batch}] Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for clean_images in val_loader:
                clean_images = clean_images.to(device)
                noisy_images = add_noise(clean_images, noise_type='gaussian', factor=NOISE_FACTOR).to(device)
                outputs = model(noisy_images)
                loss = criterion(outputs, clean_images)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # End of Epoch Summary
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{model.name()}.pth")
            print(f"Best model saved (Val Loss: {best_val_loss:.4f})")
    print(" Training Complete!")


if __name__ == "__main__":
    train()