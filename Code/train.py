import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import multiprocessing
from tqdm import tqdm  # <--- NEW IMPORT

# --- Import your modules ---
from dataset import COCODataset  # Or UnlabeledCOCODataset depending on your file
from models import *
from utils import add_noise, seed_everything

# --- Configuration ---
SEED = 42
DATA_PATH = '../data/imgs'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5
NOISE_FACTOR = 0.8
NUM_WORKERS = multiprocessing.cpu_count()


def train():
    seed_everything(SEED)
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Prepare Data
    transform = transforms.Compose([
        transforms.Resize(320),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # Handle class name difference (UnlabeledCOCODataset vs COCODataset)
    try:
        full_dataset = COCODataset(root_dir=DATA_PATH, transform=transform)
    except NameError:
        from dataset import UnlabeledCOCODataset
        full_dataset = UnlabeledCOCODataset(root_dir=DATA_PATH, transform=transform)

    total_size = len(full_dataset)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    print(f"Dataset Size: {total_size}")
    print(f"Split Config: Train={train_size}, Val={val_size}, Test={test_size}")
    print(f"Sum Check: {train_size + val_size + test_size} == {total_size}")

    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size],
                                                generator=torch.Generator().manual_seed(SEED))

    print(f'Training set size: {len(train_set)} Validation set size: {len(val_set)} Test set size: {len(test_set)}')

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
                              prefetch_factor=2,
                              persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 3. Initialize Model
    model = LightweightUNet().to(device)#UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print("Starting Training Loop...")

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0

        # NEW: Wrap loader with tqdm
        train_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        train_loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}] Train")

        for batch_idx, clean_images in train_loop:
            clean_images = clean_images.to(device)
            noisy_images = add_noise(clean_images, noise_type='gaussian', factor=NOISE_FACTOR)
            noisy_images = noisy_images.to(device)

            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, clean_images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Update bar with current loss
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0

        # NEW: Wrap loader with tqdm
        val_loop = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}] Val", leave=False)

        with torch.no_grad():
            for clean_images in val_loop:
                clean_images = clean_images.to(device)
                noisy_images = add_noise(clean_images, noise_type='gaussian', factor=NOISE_FACTOR).to(device)
                outputs = model(noisy_images)
                loss = criterion(outputs, clean_images)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # End of Epoch Summary
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Fixed filename to avoid crash
            torch.save(model.state_dict(), f"{model.name()}.pth")
            print(f"💾 Best model saved (Val Loss: {best_val_loss:.4f})")

    print("Training Complete!")


if __name__ == "__main__":
    train()