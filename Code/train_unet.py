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

# --- CONFIGURATION ---

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
CHECKPOINT_DIR = "checkpoints_curriculum"
TRAIN_DIR = "./coco_data/train2017"
VAL_DIR = "./coco_data/val2017"

CURRICULUM = [
    {
        "phase_name": "Phase 1: Initial Training",
        "start_epoch": 1,
        "end_epoch": 5,
        "lr": 2e-4,
        "loss_type": "L1",
        "alpha": 0.0
    },
    {
        "phase_name": "Phase 2: Fine-Tuning",
        "start_epoch": 6,
        "end_epoch": 15,
        "lr": 5e-5,
        "loss_type": "L1",
        "alpha": 0.0
    },
    {
        "phase_name": "Phase 3: Refined Training",
        "start_epoch": 16,
        "end_epoch": 30,
        "lr": 2e-5,
        "loss_type": "L1",
        "alpha": 0.0
    },
    {
        "phase_name": "Phase 4: Vibrant Training",
        "start_epoch": 31,
        "end_epoch": 35,
        "lr": 2e-5,
        "loss_type": "Weighted",
        "alpha": 2.0
    },
    {
        "phase_name": "Phase 5: Aggressive Training",
        "start_epoch": 36,
        "end_epoch": 50,
        "lr": 1e-5,
        "loss_type": "Exponential",
        "alpha": 3.0
    }
]


# --- METRICS UTILS ---

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


# --- CUSTOM LOSS FACTORY ---

def get_loss(output, target, loss_type, alpha):
    """
    Selects the loss logic based on the training phase.
    """
    abs_diff = torch.abs(output - target)

    # 1. Standard L1 (Phases 1-3)
    if loss_type == "L1":
        return abs_diff.mean()

    a = target[:, 0, :, :]
    b = target[:, 1, :, :]
    saturation = torch.sqrt(a ** 2 + b ** 2)

    # 2. Linear Weighting (Phase 4: Vibrant)
    if loss_type == "Weighted":
        weights = 1.0 + (saturation * alpha)
        weights = weights.unsqueeze(1).expand_as(abs_diff)
        return (abs_diff * weights).mean()

    # 3. Exponential Weighting (Phase 5: Aggressive)
    elif loss_type == "Exponential":
        weights = torch.exp(alpha * saturation)
        weights = weights.unsqueeze(1).expand_as(abs_diff)
        return (abs_diff * weights).mean()

    return abs_diff.mean()


# --- TRAINING LOOP ---
def train_fn(loader, model, optimizer, scaler, ssim_metric, loss_type, alpha):
    model.train()
    loop = tqdm(loader, leave=True)
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()

    for idx, (L, ab) in enumerate(loop):
        L, ab = L.to(DEVICE), ab.to(DEVICE)

        with torch.amp.autocast('cuda'):
            output = model(L)
            loss = get_loss(output, ab, loss_type, alpha)

        psnr = calculate_psnr(output.detach(), ab)

        loss_meter.update(loss.item(), L.size(0))
        psnr_meter.update(psnr.item(), L.size(0))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss_meter.avg, psnr=f"{psnr_meter.avg:.2f}")

    return loss_meter.avg


# --- VALIDATION LOOP ---
def validate_fn(loader, model, ssim_metric):
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    with torch.no_grad():
        for L, ab in loader:
            L, ab = L.to(DEVICE), ab.to(DEVICE)
            output = model(L)

            psnr = calculate_psnr(output, ab)
            ssim = ssim_metric(output, ab)

            psnr_meter.update(psnr.item(), L.size(0))
            ssim_meter.update(ssim.item(), L.size(0))

    return psnr_meter.avg, ssim_meter.avg


# --- MAIN EXECUTION ---
def main():
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)

    print(f" Starting Training")

    model = UNet().to(DEVICE)
    scaler = torch.amp.GradScaler('cuda')
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(DEVICE)

    train_ds = ColorizationDataset(root_dir=TRAIN_DIR, split='train')
    val_ds = ColorizationDataset(root_dir=VAL_DIR, split='val')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    best_ssim = 0.0

    for phase in CURRICULUM:
        print(f"\n{'=' * 40}")
        print(f"🎓 {phase['phase_name']}")
        print(f"   Epochs: {phase['start_epoch']} - {phase['end_epoch']}")
        print(f"   LR: {phase['lr']} | Loss: {phase['loss_type']}")
        print(f"{'=' * 40}")

        optimizer = optim.Adam(model.parameters(), lr=phase['lr'])

        for epoch in range(phase['start_epoch'], phase['end_epoch'] + 1):
            print(f"Epoch {epoch}")

            train_loss = train_fn(train_loader, model, optimizer, scaler, ssim_metric,
                                  phase['loss_type'], phase['alpha'])

            val_psnr, val_ssim = validate_fn(val_loader, model, ssim_metric)

            print(f"    Loss: {train_loss:.4f} |  Val PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.3f}")

            if val_ssim > best_ssim:
                best_ssim = val_ssim
                torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model_curriculum.pth")

            with torch.no_grad():
                val_L, val_ab = next(iter(val_loader))
                val_L, val_ab = val_L.to(DEVICE), val_ab.to(DEVICE)
                val_pred = model(val_L)
                save_results(val_L, val_ab, val_pred, epoch, folder=CHECKPOINT_DIR)

    print("\n Model Training on 50-Epochs Completed")


if __name__ == "__main__":
    main()