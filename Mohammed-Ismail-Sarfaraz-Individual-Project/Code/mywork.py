# ------------------------------------- Utilities ---------------------------------------------

import numpy as np
from PIL import Image
from skimage import color
import torch
import matplotlib.pyplot as plt
import os

def preprocess_image(pil_image, transform=None):
    """
    The Master Decolorizer: Converts a PIL RGB image into L and ab tensors.
    """

    # 1. Apply PyTorch transforms
    if transform:
        pil_image = transform(pil_image)

    # 2. Convert to Numpy
    img_np = np.array(pil_image)

    # 3. Convert RGB to Lab
    img_lab = color.rgb2lab(img_np).astype("float32")

    # 4. Normalize to [-1, 1] range
    img_lab[:, :, 0] = (img_lab[:, :, 0] / 50.0) - 1.0
    img_lab[:, :, 1:] = (img_lab[:, :, 1:] / 128.0)

    # 5. Convert to Tensor
    img_tensor = torch.from_numpy(img_lab.transpose((2, 0, 1)))

    # 6. Split into Input (L) and Target (ab)
    L = img_tensor[[0], ...]
    ab = img_tensor[[1, 2], ...]

    return L, ab


def lab_to_rgb(L_tensor, ab_tensor):
    """
    Converts a batch of L and ab tensors back to RGB numpy images.
    """
    rgb_imgs = []

    L = L_tensor.detach().cpu().numpy()
    ab = ab_tensor.detach().cpu().numpy()

    batch_size = L.shape[0]

    for i in range(batch_size):
        # 1. Denormalize
        img_l = (L[i, 0, :, :] + 1.0) * 50.0
        img_ab = ab[i, :, :, :] * 128.0

        # 2. Stack L and ab channels
        img_ab = img_ab.transpose((1, 2, 0))
        lab_img = np.zeros((img_l.shape[0], img_l.shape[1], 3))
        lab_img[:, :, 0] = img_l
        lab_img[:, :, 1:] = img_ab

        # 3. Convert Lab to RGB
        rgb_img = color.lab2rgb(lab_img)
        rgb_imgs.append(rgb_img)

    return rgb_imgs


def visualize_comparison(L_input, ab_input, ab_pred, save_path=None):
    """
    Displays: Grayscale Input | Ground Truth Color | Predicted Color
    """
    real_rgb_batch = lab_to_rgb(L_input, ab_input)
    fake_rgb_batch = lab_to_rgb(L_input, ab_pred)

    # Visualize
    img_l = L_input[0, 0].detach().cpu().numpy()
    img_real = real_rgb_batch[0]
    img_fake = fake_rgb_batch[0]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # 1. The "Decolored" Version (Input)
    ax[0].imshow(img_l, cmap='gray')
    ax[0].set_title("Input (Grayscale)")
    ax[0].axis("off")

    # 2. The Ground Truth (Target)
    ax[1].imshow(img_real)
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    # 3. The Model Output
    ax[2].imshow(img_fake)
    ax[2].set_title("Model Prediction")
    ax[2].axis("off")

    if save_path:
        plt.savefig(save_path)
        print(f" Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()

# --- WRAPPER FUNCTION FOR BACKWARD COMPATIBILITY ---

def save_results(L, ab_input, ab_output, epoch, folder="results"):

    if not os.path.exists(folder):
        os.makedirs(folder)

    path = f"{folder}/epoch_{epoch}.png"
    visualize_comparison(L, ab_input, ab_output, save_path=path)

# ------------------------------------- Models ---------------------------------------------

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.e1 = conv_block(1, 64)
        self.e2 = conv_block(64, 128)
        self.e3 = conv_block(128, 256)
        self.e4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.b = conv_block(512, 1024)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d1 = conv_block(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d2 = conv_block(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d3 = conv_block(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d4 = conv_block(128, 64)

        # Output
        self.out = nn.Conv2d(64, 2, kernel_size=1)
        self.tanh = nn.Tanh()
    def forward(self, x):
        # Encoder
        c1 = self.e1(x)
        p1 = self.pool(c1)

        c2 = self.e2(p1)
        p2 = self.pool(c2)

        c3 = self.e3(p2)
        p3 = self.pool(c3)

        c4 = self.e4(p3)
        p4 = self.pool(c4)

        # Bottleneck
        b = self.b(p4)

        # Decoder (with Skip Connections)
        u1 = self.up1(b)
        cat1 = torch.cat((u1, c4), dim=1)
        dec1 = self.d1(cat1)

        u2 = self.up2(dec1)
        cat2 = torch.cat((u2, c3), dim=1)
        dec2 = self.d2(cat2)

        u3 = self.up3(dec2)
        cat3 = torch.cat((u3, c2), dim=1)
        dec3 = self.d3(cat3)

        u4 = self.up4(dec3)
        cat4 = torch.cat((u4, c1), dim=1)
        dec4 = self.d4(cat4)

        return self.tanh(self.out(dec4))


class ResNetUNet(nn.Module):
    """
    ResNet-18 Encoder + U-Net Decoder.
    """

    def __init__(self, n_classes=2):
        super().__init__()

        # 1. Pre-trained ResNet18
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base_layers = list(base_model.children())

        # 2.Encoder Layers
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        # 3. Decoder Layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up1 = self.conv_block(512 + 256, 256)
        self.conv_up2 = self.conv_block(256 + 128, 128)
        self.conv_up3 = self.conv_block(128 + 64, 64)
        self.conv_up4 = self.conv_block(64 + 64, 64)

        # 4. Final Output
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(64, n_classes, 1)
        self.tanh = nn.Tanh()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        # --- ENCODER ---
        x = input.repeat(1, 3, 1, 1)

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)  # Bottleneck

        # --- DECODER ---

        x = self.upsample(layer4)
        if x.shape != layer3.shape:
            x = F.interpolate(x, size=layer3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        if x.shape != layer2.shape:
            x = F.interpolate(x, size=layer2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        if x.shape != layer1.shape:
            x = F.interpolate(x, size=layer1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        if x.shape != layer0.shape:
            x = F.interpolate(x, size=layer0.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up4(x)

        x = self.final_upsample(x)

        out = self.conv_last(x)
        return self.tanh(out)

# ------------------------------------- Train UNet ---------------------------------------------

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


# ------------------------------------- Test UNet ---------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import random
import os
from tqdm import tqdm

from dataset import ColorizationDataset
from models import UNet
from utils import visualize_comparison

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_DIR = "./coco_data/val2017"
MODEL_PATH = "best_model_aggressive.pth"
SAVE_DIR = "test_results"


def main():
    print(" Starting Evaluation...")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    model = UNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(" Model loaded.")
    else:
        print(f"Error: Model {MODEL_PATH} not found.")
        return

    model.eval()

    val_ds = ColorizationDataset(root_dir=VAL_DIR, split='val')

    # --- PART 1: CALCULATE AVERAGE LOSS ---
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    loss_fn = nn.L1Loss()
    total_loss = 0

    print("Calculating Average Loss over entire validation set...")
    with torch.no_grad():
        for L, ab in tqdm(val_loader):
            L = L.to(DEVICE)
            ab = ab.to(DEVICE)

            output = model(L)

            loss = loss_fn(output, ab)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f" Average Validation L1 Loss: {avg_loss:.4f}")

    # --- PART 2: RANDOM VISUALIZATION  ---
    print("\nGenerating random visualizations...")

    total_images = len(val_ds)
    random_indices = random.sample(range(total_images), 3)
    print(f"Selected Random Indices: {random_indices}")

    subset_ds = Subset(val_ds, random_indices)
    subset_loader = DataLoader(subset_ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (L, ab) in enumerate(subset_loader):
            L = L.to(DEVICE)
            ab = ab.to(DEVICE)

            output = model(L)

            save_path = f"{SAVE_DIR}/random_test_{i + 1}.png"
            visualize_comparison(
                L,
                ab,
                output,
                save_path=save_path
            )
            print(f"   -> Saved: {save_path}")

    print("\nDone! Check the 'test_results' folder.")


if __name__ == "__main__":
    main()

# ------------------------------------- Train ResNet-UNet ---------------------------------------------

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

# ------------------------------------- Test ResNet-UNet ---------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import random
import os
from tqdm import tqdm

from dataset import ColorizationDataset
from models import ResNetUNet
from utils import visualize_comparison

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_DIR = "./coco_data/val2017"
MODEL_PATH = "best_resnet_model.pth"
SAVE_DIR = "test_results_resnet"


def main():
    print(" Starting ResNet Evaluation...")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    model = ResNetUNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded.")
    else:
        print(f" Error: Model {MODEL_PATH} not found.")
        return

    model.eval()

    val_ds = ColorizationDataset(root_dir=VAL_DIR, split='val')
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    loss_fn = nn.L1Loss()
    total_loss = 0

    print(" Calculating metrics...")
    with torch.no_grad():
        for L, ab in tqdm(val_loader):
            L, ab = L.to(DEVICE), ab.to(DEVICE)
            output = model(L)
            loss = loss_fn(output, ab)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f" Average Validation L1 Loss: {avg_loss:.4f}")

    print("\n Generating visualizations...")
    random_indices = random.sample(range(len(val_ds)), 3)
    subset_ds = Subset(val_ds, random_indices)
    subset_loader = DataLoader(subset_ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, (L, ab) in enumerate(subset_loader):
            L, ab = L.to(DEVICE), ab.to(DEVICE)
            output = model(L)
            save_path = f"{SAVE_DIR}/random_test_{i + 1}.png"
            visualize_comparison(L, ab, output, save_path=save_path)
            print(f"   -> Saved: {save_path}")

if __name__ == "__main__":
    main()