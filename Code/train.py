import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import multiprocessing
from tqdm import tqdm
import random

# --- Modules ---
from dataset import COCODataset # Ensure this matches your file
# from models import LightweightUNet as UNet #Ensure this matches your models file
from models import Denoising_Model as UNet
from utils import add_noise, seed_everything, plot_training_metrics, CombinedLoss, calculate_psnr

# --- Configuration ---
SEED = 42
DATA_PATH = '../data/imgs'
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50 # Set high, Early Stopping will stop it when ready
NOISE_FACTOR = 0.5
NUM_WORKERS = multiprocessing.cpu_count()

# --- Hyperparameters ---
SCHEDULER_PATIENCE = 2 # Reduce LR if no improvement for 2 epochs
SCHEDULER_FACTOR = 0.5 #
EARLY_STOPPING_PATIENCE = 5 # Stop training if no improvement for 5 epochs

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # 2. Prepare Data
    transform = transforms.Compose([
        transforms.Resize(320),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


    full_dataset = COCODataset(root_dir=DATA_PATH, transform=transform)

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    if train_size + val_size + test_size != total_size:
        raise ValueError("Math Error: Splits do not sum to total!")

    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    # 3. Initialize Model
    model = UNet().to(device)
    criterion = CombinedLoss(alpha=0.85)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Schedulers & Stoppers ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=0.0001)

    print("Starting Training...")
    best_val_loss = float("inf")

    train_loss_history = []
    val_loss_history = []
    val_psnr_history = []

    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()

        # --- TRAIN ---
        model.train()
        train_loss = 0.0

        train_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        train_loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}] Train")

        for batch_idx, clean_images in train_loop:
            clean_images = clean_images.to(device)
            current_noise_type = random.choice(['gaussian', 'salt_pepper'])
            noisy_images = add_noise(clean_images, noise_type=current_noise_type, factor=NOISE_FACTOR).to(device)

            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, clean_images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0

        val_loop = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}] Val", leave=False)

        with torch.no_grad():
            for clean_images in val_loop:
                clean_images = clean_images.to(device)
                current_noise_type = random.choice(['gaussian', 'salt_pepper'])
                noisy_images = add_noise(clean_images, noise_type=current_noise_type, factor=NOISE_FACTOR).to(device)

                outputs = model(noisy_images)

                loss = criterion(outputs, clean_images)
                val_loss += loss.item()

                batch_psnr = calculate_psnr(outputs, clean_images)
                val_psnr += batch_psnr.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_psnr = val_psnr / len(val_loader)

        val_loss_history.append(avg_val_loss)
        val_psnr_history.append(avg_val_psnr)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val PSNR: {avg_val_psnr:.2f} dB")

        # --- SCHEDULER STEP ---
        scheduler.step(avg_val_loss)

        # --- SAVE BEST MODEL ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{model.name()}.pth")
            print(f"Best model saved.")

        # --- EARLY STOPPING CHECK ---
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered. Training stopped.")
            break

    plot_training_metrics(train_loss_history, val_loss_history, val_psnr_history)
    print(" Training Complete!")


if __name__ == "__main__":
    train()