import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure

from dataset import ColorizationDataset
from models import ResNetUNet
from utils import save_results

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (Color Boosting)
ALPHA = 3.0

TRAIN_DIR = "./coco_data/train2017"
VAL_DIR = "./coco_data/val2017"
CHECKPOINT_DIR = "checkpoints_resnet"

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


# --- CUSTOM LOSS (Exponential) ---
def exponential_l1_loss(output, target):
    abs_diff = torch.abs(output - target)
    a = target[:, 0, :, :]
    b = target[:, 1, :, :]
    saturation = torch.sqrt(a ** 2 + b ** 2)
    weights = torch.exp(ALPHA * saturation)
    weights = weights.unsqueeze(1).expand_as(abs_diff)
    loss = (abs_diff * weights).mean()
    return loss


# --- TRAINING LOOP ---
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
            loss = exponential_l1_loss(output, ab)

        psnr = calculate_psnr(output.detach(), ab)
        ssim = ssim_metric(output.detach(), ab)

        loss_meter.update(loss.item(), L.size(0))
        psnr_meter.update(psnr.item(), L.size(0))
        ssim_meter.update(ssim.item(), L.size(0))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss_meter.avg, psnr=f"{psnr_meter.avg:.2f}", ssim=f"{ssim_meter.avg:.3f}")

    return loss_meter.avg, psnr_meter.avg, ssim_meter.avg


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
            loss = nn.functional.l1_loss(output, ab)

            psnr = calculate_psnr(output, ab)
            ssim = ssim_metric(output, ab)

            loss_meter.update(loss.item(), L.size(0))
            psnr_meter.update(psnr.item(), L.size(0))
            ssim_meter.update(ssim.item(), L.size(0))

    return loss_meter.avg, psnr_meter.avg, ssim_meter.avg


def main():
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

    print(f" Starting ResNet-UNet Training on: {DEVICE}")

    model = ResNetUNet().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(DEVICE)

    global TRAIN_DIR
    if not os.path.exists(TRAIN_DIR):
        if os.path.exists(VAL_DIR):
            TRAIN_DIR = VAL_DIR
        else:
            return

    train_ds = ColorizationDataset(root_dir=TRAIN_DIR, split='train')
    val_ds = ColorizationDataset(root_dir=VAL_DIR, split='val')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    best_val_score = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")

        train_loss, train_psnr, train_ssim = train_fn(train_loader, model, optimizer, scaler, ssim_metric)
        val_loss, val_psnr, val_ssim = validate_fn(val_loader, model, ssim_metric)

        print(f"\tTrain Loss: {train_loss:.4f}")
        print(f"\tVal PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.3f}")

        if val_ssim > best_val_score:
            best_val_score = val_ssim
            torch.save(model.state_dict(), "best_resnet_model.pth")
            print("\t New Best ResNet Model Saved!")

        with torch.no_grad():
            val_L, val_ab = next(iter(val_loader))
            val_L, val_ab = val_L.to(DEVICE), val_ab.to(DEVICE)
            val_pred = model(val_L)
            save_results(val_L, val_ab, val_pred, epoch, folder=CHECKPOINT_DIR)

    print(" ResNet Training Complete!")

if __name__ == "__main__":
    main()