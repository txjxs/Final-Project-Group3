import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure

# Import your modules
from dataset import ColorizationDataset
from model import UNet
from utils import save_results

# --- CONFIG ---
LEARNING_RATE = 2e-5  # Low LR for fine-tuning
BATCH_SIZE = 32
START_EPOCH = 31  # Continue from where you left off
ADDITIONAL_EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Color Weighting Factor
# Higher = More vibrant, but maybe more artifacts. 1.5 - 2.0 is a good sweet spot.
SATURATION_WEIGHT = 2.0

# Paths
LOAD_MODEL_PATH = "best_model_refined.pth"
TRAIN_DIR = "./coco_data/train2017"
VAL_DIR = "./coco_data/val2017"
CHECKPOINT_DIR = "checkpoints"


# --- METRIC UTILS ---
class AverageMeter:
    def __init__(self): self.reset()

    def reset(self): self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_psnr(output, target):
    mse = torch.mean((output - target) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    return torch.mean(10 * torch.log10((2.0 ** 2) / mse))


# --- CUSTOM WEIGHTED LOSS ---
def weighted_l1_loss(output, target):
    """
    L1 Loss that penalizes errors on colorful pixels MORE than gray pixels.
    """
    # 1. Calculate L1 Error per pixel
    abs_diff = torch.abs(output - target)

    # 2. Calculate Saturation of the True Image (Ground Truth)
    # a channel is index 0, b is index 1
    # Saturation is roughly magnitude of the ab vector
    a = target[:, 0, :, :]
    b = target[:, 1, :, :]
    saturation = torch.sqrt(a ** 2 + b ** 2)  # Shape: (B, H, W)

    # 3. Create Weight Map
    # Base weight is 1.0. Add weight proportional to saturation.
    # If pixel is gray (sat=0), weight is 1.0
    # If pixel is vibrant (sat=1), weight is 1.0 + SATURATION_WEIGHT
    weights = 1.0 + (saturation * SATURATION_WEIGHT)

    # Expand weights to match output shape (B, 2, H, W)
    weights = weights.unsqueeze(1).expand_as(abs_diff)

    # 4. Weighted Mean
    loss = (abs_diff * weights).mean()

    return loss


# --- TRAINING FUNCTION ---
def train_fn(loader, model, optimizer, scaler, ssim_metric):
    model.train()
    loop = tqdm(loader, leave=True)

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for idx, (L, ab) in enumerate(loop):
        L, ab = L.to(DEVICE), ab.to(DEVICE)

        with torch.amp.autocast('cuda'):
            output = model(L)
            # USE CUSTOM LOSS HERE
            loss = weighted_l1_loss(output, ab)

        # Metrics
        psnr = calculate_psnr(output.detach(), ab)
        ssim = ssim_metric(output.detach(), ab)

        loss_meter.update(loss.item(), L.size(0))
        psnr_meter.update(psnr.item(), L.size(0))
        ssim_meter.update(ssim.item(), L.size(0))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss_meter.avg, psnr=f"{psnr_meter.avg:.2f}dB", ssim=f"{ssim_meter.avg:.3f}")

    return loss_meter.avg, psnr_meter.avg, ssim_meter.avg


# --- VALIDATION FUNCTION ---
def validate_fn(loader, model, ssim_metric):
    model.eval()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    loop = tqdm(loader, desc="Validating", leave=False)

    with torch.no_grad():
        for L, ab in loop:
            L, ab = L.to(DEVICE), ab.to(DEVICE)
            output = model(L)

            # Use same weighted loss for fair comparison
            loss = weighted_l1_loss(output, ab)
            psnr = calculate_psnr(output, ab)
            ssim = ssim_metric(output, ab)

            loss_meter.update(loss.item(), L.size(0))
            psnr_meter.update(psnr.item(), L.size(0))
            ssim_meter.update(ssim.item(), L.size(0))

    return loss_meter.avg, psnr_meter.avg, ssim_meter.avg


# --- MAIN ---
def main():
    global TRAIN_DIR
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

    print(f"🚀 Starting VIBRANT Training on: {DEVICE}")

    model = UNet().to(DEVICE)

    # LOAD PREVIOUS BEST MODEL
    if os.path.exists(LOAD_MODEL_PATH):
        print(f"Loading weights from {LOAD_MODEL_PATH}...")
        checkpoint = torch.load(LOAD_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print(" Weights loaded!")
    else:
        print(f"Error: {LOAD_MODEL_PATH} not found. Train normal model first.")
        return

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(DEVICE)

    if not os.path.exists(TRAIN_DIR):
        if os.path.exists(VAL_DIR):
            TRAIN_DIR = VAL_DIR
        else:
            return

    train_ds = ColorizationDataset(root_dir=TRAIN_DIR, split='train')
    val_ds = ColorizationDataset(root_dir=VAL_DIR, split='val')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    best_val_score = 0.0  # We might track SSIM or PSNR

    for epoch in range(START_EPOCH, START_EPOCH + ADDITIONAL_EPOCHS):
        print(f"\nEpoch [{epoch}/{START_EPOCH + ADDITIONAL_EPOCHS - 1}] (Vibrant Fine-Tuning)")

        train_loss, train_psnr, train_ssim = train_fn(train_loader, model, optimizer, scaler, ssim_metric)
        val_loss, val_psnr, val_ssim = validate_fn(val_loader, model, ssim_metric)

        print(f"\tTrain: Loss {train_loss:.4f} | PSNR {train_psnr:.2f} | SSIM {train_ssim:.3f}")
        print(f"\tVal:   Loss {val_loss:.4f} | PSNR {val_psnr:.2f} | SSIM {val_ssim:.3f}")

        # Save model based on SSIM (better for visual quality than PSNR)
        if val_ssim > best_val_score:
            best_val_score = val_ssim
            torch.save(model.state_dict(), "best_model_vibrant.pth")
            print("\t New Best VIBRANT Model Saved!")

        with torch.no_grad():
            val_L, val_ab = next(iter(val_loader))
            val_L, val_ab = val_L.to(DEVICE), val_ab.to(DEVICE)
            val_pred = model(val_L)
            save_results(val_L, val_ab, val_pred, epoch, folder=CHECKPOINT_DIR)

    print("Vibrant Training Complete!")


if __name__ == "__main__":
    main()