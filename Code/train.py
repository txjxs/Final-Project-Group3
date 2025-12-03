import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure

from dataset import ColorizationDataset
from models import UNet
from utils import save_results

LEARNING_RATE = 2e-4
BATCH_SIZE = 32
NUM_EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "./coco_data/train2017"
VAL_DIR = "./coco_data/val2017"
CHECKPOINT_DIR = "checkpoints"


# --- METRIC UTILS ---
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_psnr(output, target):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) on a batch.
    Assumes tensors are in range [-1, 1], so data_range = 2.0
    """
    mse = torch.mean((output - target) ** 2, dim=[1, 2, 3])  # Batch-wise MSE
    mse = torch.clamp(mse, min=1e-10)

    data_range = 2.0
    psnr = 10 * torch.log10((data_range ** 2) / mse)
    return torch.mean(psnr)


# --- TRAINING FUNCTION ---
def train_fn(loader, model, optimizer, loss_fn, scaler, ssim_metric):
    model.train()
    loop = tqdm(loader, leave=True)

    # Trackers
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for idx, (L, ab) in enumerate(loop):
        L = L.to(DEVICE)
        ab = ab.to(DEVICE)

        # Train with Mixed Precision
        with torch.amp.autocast('cuda'):
            output = model(L)
            loss = loss_fn(output, ab)

        # Metrics
        # Detach to avoid memory leaks during metric calculation
        psnr = calculate_psnr(output.detach(), ab)
        ssim = ssim_metric(output.detach(), ab)

        # Update Trackers
        loss_meter.update(loss.item(), L.size(0))
        psnr_meter.update(psnr.item(), L.size(0))
        ssim_meter.update(ssim.item(), L.size(0))

        # Backprop
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update Progress Bar
        loop.set_postfix(loss=loss_meter.avg, psnr=f"{psnr_meter.avg:.2f}dB", ssim=f"{ssim_meter.avg:.3f}")

    return loss_meter.avg, psnr_meter.avg, ssim_meter.avg


# --- VALIDATION FUNCTION ---
def validate_fn(loader, model, loss_fn, ssim_metric):
    model.eval()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    loop = tqdm(loader, desc="Validating", leave=False)

    with torch.no_grad():
        for L, ab in loop:
            L = L.to(DEVICE)
            ab = ab.to(DEVICE)

            output = model(L)
            loss = loss_fn(output, ab)
            psnr = calculate_psnr(output, ab)
            ssim = ssim_metric(output, ab)

            loss_meter.update(loss.item(), L.size(0))
            psnr_meter.update(psnr.item(), L.size(0))
            ssim_meter.update(ssim.item(), L.size(0))

    return loss_meter.avg, psnr_meter.avg, ssim_meter.avg


# --- MAIN ---
def main():
    global TRAIN_DIR
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    print(f"Training on: {DEVICE}")

    model = UNet().to(DEVICE)
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    # Initialize SSIM Metric
    # data_range is 2.0 because our tanh output is [-1, 1]
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(DEVICE)

    # Check Data
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: {TRAIN_DIR} not found.")
        # Fallback for testing if train doesn't exist
        if os.path.exists(VAL_DIR):
            print("Using Val dir as Train for testing purposes...")
            TRAIN_DIR = VAL_DIR
        else:
            return

    # Loaders
    train_ds = ColorizationDataset(root_dir=TRAIN_DIR, split='train')
    val_ds = ColorizationDataset(root_dir=VAL_DIR, split='val')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    best_val_psnr = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")

        # 1. Train
        train_loss, train_psnr, train_ssim = train_fn(train_loader, model, optimizer, loss_fn, scaler, ssim_metric)

        # 2. Validate
        val_loss, val_psnr, val_ssim = validate_fn(val_loader, model, loss_fn, ssim_metric)

        print(f"\tTrain: Loss {train_loss:.4f} | PSNR {train_psnr:.2f} | SSIM {train_ssim:.3f}")
        print(f"\tVal:   Loss {val_loss:.4f} | PSNR {val_psnr:.2f} | SSIM {val_ssim:.3f}")

        # 3. Save Best Model
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), "best_model.pth")
            print("\tNew Best Model Saved!")

        # 4. Visualize
        with torch.no_grad():
            val_L, val_ab = next(iter(val_loader))
            val_L = val_L.to(DEVICE)
            val_ab = val_ab.to(DEVICE)
            val_pred = model(val_L)
            save_results(val_L, val_ab, val_pred, epoch, folder=CHECKPOINT_DIR)

        # 5. Checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/epoch_{epoch + 1}.pth")

    print("Training Complete!")

if __name__ == "__main__":
    main()