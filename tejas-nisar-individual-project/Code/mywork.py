#Tejas Nisar Contributions
#%%
import torch
from models import LightweightUNet, HeavyUNet
from thop import profile

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare():
    print("initializing models...")
    # 1. Load Models
    heavy_model = HeavyUNet()
    light_model = LightweightUNet()

    # 2. Count Parameters
    params_heavy = count_parameters(heavy_model)
    params_light = count_parameters(light_model)

    print(f"\n📉 FAIR ARCHITECTURE COMPARISON:")
    print(f"   Equivalent Standard U-Net: {params_heavy:,} parameters")
    print(f"   Your Lightweight U-Net:    {params_light:,} parameters")
    print(f"   Reduction Factor:          {params_heavy / params_light:.1f}x smaller")

    # 3. FLOPs
    input = torch.randn(1, 3, 256, 256)
    try:
        flops_heavy, _ = profile(heavy_model, inputs=(input,), verbose=False)
        flops_light, _ = profile(light_model, inputs=(input,), verbose=False)

        print(f"\n⚡ Speed/Cost Comparison (FLOPs):")
        print(f"   Equivalent Standard U-Net: {flops_heavy / 1e9:.2f} GFLOPs")
        print(f"   Your Lightweight U-Net:    {flops_light / 1e9:.2f} GFLOPs")
        print(f"   Efficiency Gain:           {flops_heavy / flops_light:.1f}x faster")
    except:
        pass


if __name__ == "__main__":
    compare()

#%%
import os
from PIL import Image
from torch.utils.data import Dataset


class COCODataset(Dataset):
    """
    Custom class to load and preprocess COCO dataset
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(self.image_files)} images in {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_name)

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


#%%
import torch
from models import Denoising_Model

# 1. Create the model
model = Denoising_Model()

# 2. Create a fake image batch (Batch=4, Channels=3, Height=128, Width=128)
dummy_input = torch.randn(4, 3, 128, 128)
print(dummy_input)
# 3. Pass it through
output = model(dummy_input)

# 4. Check
print(f"Input Shape:  {dummy_input.shape}")
print(f"Output Shape: {output.shape}")

if dummy_input.shape == output.shape:
    print(" SUCCESS: Output shape matches Input shape!")
else:
    print("ERROR: Shapes do not match.")

#%%
import torch
from torch import nn


class Noisy_Encoder(nn.Module):
    def __init__(self):
        super(Noisy_Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
class Noisy_Decoder(nn.Module):
    def __init__(self):
        super(Noisy_Decoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()

        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1,output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Denoising_Model(nn.Module):

    @staticmethod
    def name():
        return 'Denoising_Model'

    def __init__(self):
        super(Denoising_Model, self).__init__()
        self.encoder = Noisy_Encoder()
        self.decoder = Noisy_Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class UNet(nn.Module):

    def name(self):
        return 'UNet'


    def __init__(self):
        super(UNet, self).__init__()

        # --- ENCODER (Downsampling) ---
        # 128 -> 64
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # 64 -> 32
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 32 -> 16 (Bottleneck)
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # --- DECODER (Upsampling) ---

        # Up 1: 16 -> 32
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up1_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 because we concat (64 from up1 + 64 from enc2)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Up 2: 32 -> 64
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 64 because we concat (32 from up2 + 32 from enc1)
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Up 3: 64 -> 128
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),  # No concat here, just final polish
            nn.Sigmoid()
        )

    def forward(self, x):
        # --- DOWN ---
        e1 = self.enc1(x)  # Save this for skip connection! (Shape: 32 channels)
        e2 = self.enc2(e1)  # Save this too! (Shape: 64 channels)
        latent = self.enc3(e2)  # Bottleneck (Shape: 128 channels)

        # --- UP ---

        # Un-squeeze bottleneck
        d1 = self.up1(latent)
        # SKIP CONNECTION 1: Glue d1 and e2 together
        # d1 is 64 ch, e2 is 64 ch -> Result is 128 ch
        d1 = torch.cat((d1, e2), dim=1)
        d1 = self.up1_conv(d1)

        d2 = self.up2(d1)
        # SKIP CONNECTION 2: Glue d2 and e1 together
        # d2 is 32 ch, e1 is 32 ch -> Result is 64 ch
        d2 = torch.cat((d2, e1), dim=1)
        d2 = self.up2_conv(d2)

        output = self.up3(d2)
        output = self.final_conv(output)

        return output


class DSConv(nn.Module):
    """
    Depthwise Separable Convolution Block
    1. Depthwise: Spatial filtering (groups=in_ch)
    2. Pointwise: Channel mixing (kernel_size=1)
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            # --- Depthwise ---
            # Groups=in_ch makes it depthwise (one filter per channel)
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            # --- Pointwise ---
            # 1x1 conv mixes the channels
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class DSConvDouble(nn.Module):
    """
    The lightweight equivalent of the 'DoubleConv' block.
    It stacks two DSConv blocks to learn complex features efficiently.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            DSConv(in_ch, out_ch),
            DSConv(out_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class LightweightUNet(nn.Module):
    def __init__(self):
        super(LightweightUNet, self).__init__()

        # --- ENCODER ---
        # Initial Block: Standard Conv -> DSConv
        # We use a standard conv first to mix the raw RGB channels properly
        self.inc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DSConv(32, 32)
        )

        # Downsampling path
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DSConvDouble(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DSConvDouble(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DSConvDouble(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DSConvDouble(256, 512))  # Bottleneck

        # --- DECODER ---
        # Upsampling path with skip connections
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1_conv = DSConvDouble(512 + 256, 256)  # 1024 from bottleneck + 512 from skip

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2_conv = DSConvDouble(256 + 128, 128)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3_conv = DSConvDouble(128 + 64, 64)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4_conv = DSConvDouble(64 + 32, 32)

        # Final Output Block (1x1 Conv to get 3 channels)
        self.outc = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()  # Forces output to [0, 1]
        )

    def forward(self, x):
        # Down
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Up (Concatenation happens here)
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up1_conv(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up2_conv(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up3_conv(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up4_conv(x)

        return self.outc(x)

    @staticmethod
    def name():
        return 'LightweightUNet'


class DoubleConv(nn.Module):
    """
    Standard Heavy Double Convolution
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class HeavyUNet(nn.Module):
    def __init__(self):
        super(HeavyUNet, self).__init__()

        # Exact same shape as LightweightUNet, but using Standard Convs
        self.inc = DoubleConv(3, 32)

        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1_conv = DoubleConv(512 + 256, 256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2_conv = DoubleConv(256 + 128, 128)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3_conv = DoubleConv(128 + 64, 64)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4_conv = DoubleConv(64 + 32, 32)

        self.outc = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up1_conv(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up2_conv(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up3_conv(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up4_conv(x)

        return self.outc(x)

    @staticmethod
    def name():
        return 'HeavyUNet'

#%%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from pytorch_msssim import ssim


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print(f"🔒 Seed set to {seed}")


def calculate_psnr(img1, img2):
    """
    Calculates PSNR.
    img1, img2: [Batch, 3, H, W]
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0  # Return a high value instead of inf for plotting stability
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def add_noise(img_tensor, noise_type='gaussian', factor=0.5):
    noisy_img = img_tensor.clone()
    if noise_type == 'gaussian':
        noise = torch.randn_like(img_tensor) * factor
        noisy_img = img_tensor + noise
    elif noise_type == 'salt_pepper':
        prob = torch.rand_like(img_tensor)
        noisy_img[prob < (factor / 2)] = 1.0
        noisy_img[prob > 1 - (factor / 2)] = 0.0
    else:
        raise ValueError("Invalid noise_type")
    return torch.clamp(noisy_img, 0., 1.)


def plot_training_metrics(train_losses, val_losses, val_psnrs, save_path='training_metrics.png'):
    """
    Plots Loss curves and PSNR curves side-by-side.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(15, 5))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: PSNR
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_psnrs, label='Validation PSNR', color='green', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.title('Model Improvement (PSNR)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Metrics plot saved to {save_path}")


def plot_denoising_result(original, noisy, denoised, psnr=None, save_path=None, Model_name="Unknown Model"):
    if original.dim() == 4: original = original[0]
    if noisy.dim() == 4: noisy = noisy[0]
    if denoised.dim() == 4: denoised = denoised[0]

    original = original.detach().cpu().permute(1, 2, 0).numpy()
    noisy = noisy.detach().cpu().permute(1, 2, 0).numpy()
    denoised = denoised.detach().cpu().permute(1, 2, 0).numpy()

    original = np.clip(original, 0, 1)
    noisy = np.clip(noisy, 0, 1)
    denoised = np.clip(denoised, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    if Model_name:
        fig.suptitle(f"Model: {Model_name}", fontsize=20, weight='bold')

    axes[0].imshow(original)
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(noisy)
    axes[1].set_title('Noisy', fontsize=14)
    axes[1].axis('off')

    title = "Denoised"
    if psnr is not None:
        title += f"\nPSNR: {psnr:.2f} dB"

    axes[2].imshow(denoised)
    axes[2].set_title(title, fontsize=14)
    axes[2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.85):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, output, target):
        ssim_val = ssim(output, target, data_range=1.0, size_average=True)
        return (self.alpha * (1 - ssim_val)) + ((1 - self.alpha) * self.l1(output, target))
#%%

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, dataset
from torchvision import transforms

from dataset import COCODataset
from utils import add_noise , plot_denoising_result


DATA_PATH = '../data/imgs/'

BATCH_SIZE = 4
IMG_SIZE = 128

def main():
    print(f'checking data  in {DATA_PATH} ')

    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                    transforms.ToTensor(),
                                    ])

    try:
        dataset = COCODataset(DATA_PATH, transform=transform)
    except FileNotFoundError:
        print(f'{DATA_PATH} not exist')
        return

    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    data_iter = iter(train_loader)
    clean_images = next(data_iter)

    print(f"✅ Batch Loaded Successfully!")
    print(f"   Tensor Shape: {clean_images.shape}")
    print(f"   (Batch Size, Channels, Height, Width)")


    print('adding noise...')
    noisy_img = add_noise(clean_images, noise_type='salt_pepper', factor=0.5)
    plot_denoising_result(clean_images, noisy_img)

if __name__ == '__main__':
    main()



#%%
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
EPOCHS = 50
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

#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import random  # Needed for random noise selection

# --- Modules ---
from dataset import COCODataset
# from models import LightweightUNet as UNet
# from models import HeavyUNet as UNet # Ensure we use the lightweight class
from models import Denoising_Model as UNet
from utils import add_noise, seed_everything, plot_denoising_result, calculate_psnr

# --- Config ---
SEED = 42
DATA_PATH = '../data/imgs'
MODEL_PATH = f'{UNet.name()}.pth'
# MODEL_PATH = 'Models/HeavyUNet.pth' # Matches train.py save name
BATCH_SIZE = 32
NOISE_FACTOR = 0.5


def test():
    # 1. Reproduce Setup
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Testing on device: {device}")

    # 2. Recreate Splits
    transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(256),  # Center crop for consistent testing
        transforms.ToTensor(),
    ])

    # Handle Dataset Class Name
    full_dataset = COCODataset(root_dir=DATA_PATH, transform=transform) # Now explicitly using COCODataset

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    _, _, test_set = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test Set Loaded: {len(test_set)} images")

    # 3. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f" Error: Model file '{MODEL_PATH}' not found.")
        return

    model = UNet().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(" Model weights loaded successfully.")
    except Exception as e:
        print(f" Error loading weights: {e}")
        return

    model.eval()

    # 4. Evaluate
    total_psnr = 0.0

    print("Running Evaluation...")

    # Containers for the first batch to plot later
    sample_clean, sample_noisy, sample_recon = None, None, None
    sample_noise_type = ""

    with torch.no_grad():
        for i, clean_images in enumerate(test_loader):
            clean_images = clean_images.to(device)

            # --- UPDATE: Random Noise Selection (Matches Training) ---
            current_noise_type = random.choice(['gaussian', 'salt_pepper'])

            noisy_images = add_noise(clean_images, noise_type=current_noise_type, factor=NOISE_FACTOR).to(device)

            outputs = model(noisy_images)

            # Metrics
            psnr = calculate_psnr(outputs, clean_images)
            total_psnr += psnr.item()

            # Save the first batch for visualization
            if i == 0:
                sample_clean = clean_images
                sample_noisy = noisy_images
                sample_recon = outputs
                sample_noise_type = current_noise_type

    avg_psnr = total_psnr / len(test_loader)

    print(f"\n Final Test Results:")
    print(f"   Avg PSNR: {avg_psnr:.2f} dB")

    # 5. Save Visual Report
    plot_denoising_result(
        sample_clean,
        sample_noisy,
        sample_recon,
        psnr=avg_psnr,
        save_path='test_report.png',
        Model_name=f"Heavy U-Net ({sample_noise_type} noise)"
    )


if __name__ == "__main__":
    test()

#%%
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, dataset
from torchvision import transforms

from dataset import COCODataset
from utils import add_noise , plot_denoising_result


DATA_PATH = '../data/imgs/'

BATCH_SIZE = 4
IMG_SIZE = 128

def main():
    print(f'checking data  in {DATA_PATH} ')

    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                    transforms.ToTensor(),
                                    ])

    try:
        dataset = COCODataset(DATA_PATH, transform=transform)
    except FileNotFoundError:
        print(f'{DATA_PATH} not exist')
        return

    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    data_iter = iter(train_loader)
    clean_images = next(data_iter)

    print(f"✅ Batch Loaded Successfully!")
    print(f"   Tensor Shape: {clean_images.shape}")
    print(f"   (Batch Size, Channels, Height, Width)")


    print('adding noise...')
    noisy_img = add_noise(clean_images, noise_type='salt_pepper', factor=0.5)
    plot_denoising_result(clean_images, noisy_img)

if __name__ == '__main__':
    main()

