"""
L*a*b* Color Space Dataset for Colorization

Key difference from RGB:
- Input: L* channel (lightness) - noisy grayscale
- Output: a*, b* channels (color only)
- Model focuses ONLY on predicting color, not brightness!

This is the standard approach for colorization in the literature.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from skimage import color


class LABColorizeDataset(Dataset):
    """
    L*a*b* colorization dataset

    Input: Noisy L* channel (lightness)
    Output: a*, b* channels (color)
    """

    def __init__(self, root_dir, image_size=128, noise_std=0.1):
        """
        Args:
            root_dir: Directory with images
            image_size: Size to resize to
            noise_std: Noise standard deviation for L* channel
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.noise_std = noise_std

        # Get all image files
        self.image_files = [f for f in os.listdir(root_dir)
                            if f.endswith(('.jpg', '.jpeg', '.png'))]

        print(f"Loaded {len(self.image_files)} images from {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))

        rgb = np.array(image).astype(np.float32) / 255.0

        lab = color.rgb2lab(rgb)

        L = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]

        L_norm = L / 100.0
        a_norm = (a + 128.0) / 255.0
        b_norm = (b + 128.0) / 255.0

        noise = np.random.randn(*L_norm.shape) * self.noise_std
        L_noisy = np.clip(L_norm + noise, 0, 1)

        L_noisy = torch.from_numpy(L_noisy).unsqueeze(0).float()
        ab = torch.from_numpy(np.stack([a_norm, b_norm], axis=0)).float()
        L_clean = torch.from_numpy(L_norm).unsqueeze(0).float()
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()

        return {
            'input': L_noisy,
            'target': ab,
            'L_clean': L_clean,
            'target_rgb': rgb_tensor,
        }


def lab_to_rgb(L, ab):
    """
    Convert L*a*b* tensors back to RGB

    Args:
        L: (B, 1, H, W) tensor, normalized [0, 1]
        ab: (B, 2, H, W) tensor, normalized [0, 1]

    Returns:
        rgb: (B, 3, H, W) tensor [0, 1]
    """
    L_denorm = L * 100.0  # [0, 100]
    a_denorm = ab[:, 0:1] * 255.0 - 128.0  # [-128, 127]
    b_denorm = ab[:, 1:2] * 255.0 - 128.0  # [-128, 127]

    lab = torch.cat([L_denorm, a_denorm, b_denorm], dim=1)  # (B, 3, H, W)

    lab_np = lab.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 3)

    rgb_list = []
    for i in range(lab_np.shape[0]):
        rgb = color.lab2rgb(lab_np[i])
        rgb = np.clip(rgb, 0, 1)
        rgb_list.append(rgb)

    rgb_np = np.stack(rgb_list, axis=0)  # (B, H, W, 3)
    rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()  # (B, 3, H, W)

    return rgb_tensor


def create_lab_dataloaders(train_dir, val_dir, batch_size=32,
                           image_size=128, noise_std=0.1, num_workers=4):
    """
    Create L*a*b* train and validation dataloaders
    """
    print("=" * 70)
    print("Creating L*a*b* Colorization Datasets")
    print("=" * 70)
    print("Task: Noisy L* → a*, b* (Color Only)")
    print()

    train_dataset = LABColorizeDataset(
        root_dir=train_dir,
        image_size=image_size,
        noise_std=noise_std
    )

    val_dataset = LABColorizeDataset(
        root_dir=val_dir,
        image_size=image_size,
        noise_std=noise_std
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# Test code
if __name__ == "__main__":
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("Testing Dataset")
    print("=" * 70)

    data_dir = "../data/val2017"

    dataset = LABColorizeDataset(
        root_dir=data_dir,
        image_size=128,
        noise_std=0.1
    )

    print(f"\nDataset size: {len(dataset)}")

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for i in range(3):
        sample = dataset[i]

        L_noisy = sample['input']
        ab_target = sample['target']
        L_clean = sample['L_clean']

        axes[i, 0].imshow(L_noisy.squeeze(0), cmap='gray')
        axes[i, 0].set_title('Input: Noisy L*')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(ab_target[0], cmap='RdYlGn')
        axes[i, 1].set_title('Target: a* (green-red)')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(ab_target[1], cmap='YlGnBu')
        axes[i, 2].set_title('Target: b* (blue-yellow)')
        axes[i, 2].axis('off')

        L_batch = L_clean.unsqueeze(0)
        ab_batch = ab_target.unsqueeze(0)
        rgb = lab_to_rgb(L_batch, ab_batch)

        rgb_img = rgb.squeeze(0).permute(1, 2, 0).numpy()
        axes[i, 3].imshow(rgb_img)
        axes[i, 3].set_title('Reconstructed RGB')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig('lab_dataset_test.png', dpi=150)
    print("\nSaved lab_dataset_test.png")

"""
Simple CVAE with U-Net architecture
Input: Noisy grayscale (1 channel)
Output: Clean RGB (3 channels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetEncoder(nn.Module):
    """
    U-Net Encoder with skip connections
    Processes noisy grayscale → latent space
    """

    def __init__(self, in_channels=1, latent_dim=128):
        super(UNetEncoder, self).__init__()

        self.latent_dim = latent_dim

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.flatten_size = 512 * 8 * 8
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        # Encode with skip connections
        e1 = self.enc1(x)  # (B, 64, 64, 64)
        e2 = self.enc2(e1)  # (B, 128, 32, 32)
        e3 = self.enc3(e2)  # (B, 256, 16, 16)
        e4 = self.enc4(e3)  # (B, 512, 8, 8)

        e4_flat = e4.view(e4.size(0), -1)
        mu = self.fc_mu(e4_flat)
        logvar = self.fc_logvar(e4_flat)

        skip_connections = [e1, e2, e3, e4]
        return mu, logvar, skip_connections


class UNetDecoder(nn.Module):
    """
    U-Net Decoder with skip connections
    Processes latent space → clean RGB
    """

    def __init__(self, latent_dim=128, out_channels=3):
        super(UNetDecoder, self).__init__()

        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512 + 512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, z, skip_connections):
        e1, e2, e3, e4 = skip_connections

        x = self.fc(z)
        x = x.view(x.size(0), 512, 8, 8)

        x = torch.cat([x, e4], dim=1)
        x = self.dec4(x)

        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x)

        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)

        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)

        return x


class CVAE(nn.Module):
    """
    CVAE: Noisy Grayscale → Clean RGB
    Uses U-Net architecture with skip connections
    """

    def __init__(self, in_channels=1, out_channels=3, latent_dim=128):
        super(CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = UNetEncoder(in_channels, latent_dim)
        self.decoder = UNetDecoder(latent_dim, out_channels)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        # Clamp for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)

        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon

        return z

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Noisy grayscale input (B, 1, H, W)

        Returns:
            reconstruction: Clean RGB output (B, 3, H, W)
            mu: Latent mean (B, latent_dim)
            logvar: Latent log variance (B, latent_dim)
        """
        # Encode
        mu, logvar, skip_connections = self.encoder(x)

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Decode
        reconstruction = self.decoder(z, skip_connections)

        return reconstruction, mu, logvar

    def sample(self, num_samples, device):
        """
        Generate samples from prior N(0, 1)
        Note: No skip connections available when sampling
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)

        # Create dummy skip connections
        with torch.no_grad():
            skip_connections = [
                torch.zeros(num_samples, 64, 64, 64).to(device),
                torch.zeros(num_samples, 128, 32, 32).to(device),
                torch.zeros(num_samples, 256, 16, 16).to(device),
                torch.zeros(num_samples, 512, 8, 8).to(device)
            ]
            samples = self.decoder(z, skip_connections)

        return samples


# def vae_loss(reconstruction, target, mu, logvar, beta=1.0):
#     """
#     VAE Loss = Reconstruction Loss + β * KL Divergence
#
#     Args:
#         reconstruction: Model output (B, 3, H, W)
#         target: Ground truth (B, 3, H, W)
#         mu: Latent mean (B, latent_dim)
#         logvar: Latent log variance (B, latent_dim)
#         beta: Weight for KL term
#
#     Returns:
#         total_loss, recon_loss, kl_loss
#     """
#     recon_loss = F.mse_loss(reconstruction, target, reduction='mean')
#
#     kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
#     kl_loss = torch.mean(kl_per_sample)
#
#     # Total loss
#     total_loss = recon_loss + beta * kl_loss
#
#     return total_loss, recon_loss, kl_loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class LABChromaticWeightedLoss(nn.Module):
    """
    MSE Loss weighted by color saturation in L*a*b* space
    In L*a*b*, saturation = sqrt(a^2 + b^2)
    - Gray: a=0, b=0, saturation=0
    - Vibrant: large |a| or |b|, saturation > 0
    """

    def __init__(self, chromatic_weight=3.0, base_weight=1.0):
        """
        Args:
            chromatic_weight: How much to emphasize saturated colors
            base_weight: Base weight for all regions
        """
        super(LABChromaticWeightedLoss, self).__init__()
        self.chromatic_weight = chromatic_weight
        self.base_weight = base_weight

    def compute_saturation(self, ab):
        """
        Compute color saturation from a*, b* channels

        Args:
            ab: (B, 2, H, W) - a*, b* channels normalized to [0, 1]

        Returns:
            saturation: (B, 1, H, W) - [0, 1] range
        """
        a = ab[:, 0:1] * 255.0 - 128.0
        b = ab[:, 1:2] * 255.0 - 128.0

        saturation = torch.sqrt(a**2 + b**2)

        saturation_norm = saturation / 180.0
        saturation_norm = torch.clamp(saturation_norm, 0, 1)

        return saturation_norm

    def forward(self, output, target):
        """
        Compute chromatic-weighted MSE on a*b* channels

        Args:
            output: (B, 2, H, W) predicted a*, b* in [0, 1]
            target: (B, 2, H, W) target a*, b* in [0, 1]

        Returns:
            weighted_mse: scalar loss
        """
        saturation = self.compute_saturation(target)

        weight = self.base_weight + self.chromatic_weight * saturation

        mse_per_pixel = (output - target) ** 2

        weighted_mse_per_pixel = mse_per_pixel * weight

        loss = weighted_mse_per_pixel.mean()

        return loss


class LABCVAELoss(nn.Module):
    """
    Complete CVAE loss for L*a*b* colorization

    Total loss = Chromatic-weighted MSE(a*b*) + beta * KL
    """

    def __init__(self, chromatic_weight=3.0, beta=0.001):
        super(LABCVAELoss, self).__init__()

        self.chromatic_loss = LABChromaticWeightedLoss(chromatic_weight=chromatic_weight)
        self.beta = beta

    def forward(self, output, target, mu, logvar):
        """
        Compute total CVAE loss for L*a*b*

        Args:
            output: (B, 2, H, W) predicted a*, b*
            target: (B, 2, H, W) target a*, b*
            mu: (B, latent_dim)
            logvar: (B, latent_dim)

        Returns:
            total_loss, color_loss, kl_loss
        """
        # Chromatic-weighted color loss
        color_loss = self.chromatic_loss(output, target)

        # KL divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_divergence)

        # Total loss
        total_loss = color_loss + self.beta * kl_loss

        return total_loss, color_loss, kl_loss


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from VAE_data import LABColorizeDataset, lab_to_rgb
from VAE_model import CVAE
from VAE_loss import LABCVAELoss


def train_epoch(model, dataloader, loss_fn, optimizer, device, beta_annealing=1.0):
    model.train()

    total_loss_sum = 0
    color_loss_sum = 0
    kl_loss_sum = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        L_noisy = batch['input'].to(device)  # (B, 1, H, W)
        ab_target = batch['target'].to(device)  # (B, 2, H, W)

        ab_pred, mu, logvar = model(L_noisy)

        total_loss, color_loss, kl_loss = loss_fn(ab_pred, ab_target, mu, logvar)

        if beta_annealing < 1.0:
            total_loss = color_loss + (beta_annealing * loss_fn.beta) * kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_sum += total_loss.item()
        color_loss_sum += color_loss.item()
        kl_loss_sum += kl_loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'color': f'{color_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}'
        })

    return (total_loss_sum / num_batches,
            color_loss_sum / num_batches,
            kl_loss_sum / num_batches)


def validate(model, dataloader, loss_fn, device):
    """Validate the model"""
    model.eval()

    total_loss_sum = 0
    color_loss_sum = 0
    kl_loss_sum = 0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            L_noisy = batch['input'].to(device)
            ab_target = batch['target'].to(device)

            # Forward pass
            ab_pred, mu, logvar = model(L_noisy)

            # Compute loss
            total_loss, color_loss, kl_loss = loss_fn(ab_pred, ab_target, mu, logvar)

            # Accumulate
            total_loss_sum += total_loss.item()
            color_loss_sum += color_loss.item()
            kl_loss_sum += kl_loss.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'color': f'{color_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })

    return (total_loss_sum / num_batches,
            color_loss_sum / num_batches,
            kl_loss_sum / num_batches)


def plot_training_curves(train_history, val_history, save_path):
    """Plot and save training curves"""
    epochs = range(1, len(train_history['total']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('L*a*b* Training with Chromatic Weighting', fontsize=16)

    # Total loss
    axes[0, 0].plot(epochs, train_history['total'], 'b-', label='Train')
    axes[0, 0].plot(epochs, val_history['total'], 'r-', label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Color loss
    axes[0, 1].plot(epochs, train_history['color'], 'b-', label='Train')
    axes[0, 1].plot(epochs, val_history['color'], 'r-', label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Color Loss (Chromatic-weighted MSE)')
    axes[0, 1].set_title('Color Loss on a*b* channels')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # KL loss
    axes[1, 0].plot(epochs, train_history['kl'], 'b-', label='Train')
    axes[1, 0].plot(epochs, val_history['kl'], 'r-', label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Color loss (zoomed)
    axes[1, 1].plot(epochs, train_history['color'], 'b-', label='Train')
    axes[1, 1].plot(epochs, val_history['color'], 'r-', label='Validation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Color Loss')
    axes[1, 1].set_title('Color Loss (Zoomed)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, max(train_history['color'][:5])])  # Zoom to first 5 epochs range

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_results(model, dataloader, device, save_path, num_samples=4):
    """Visualize L*a*b* colorization results"""
    model.eval()

    # Get a batch
    batch = next(iter(dataloader))
    L_noisy = batch['input'][:num_samples].to(device)
    ab_target = batch['target'][:num_samples].to(device)
    L_clean = batch['L_clean'][:num_samples].to(device)

    with torch.no_grad():
        ab_pred, _, _ = model(L_noisy)

    rgb_pred = lab_to_rgb(L_clean, ab_pred)
    rgb_target = lab_to_rgb(L_clean, ab_target)

    L_noisy = L_noisy.cpu()
    rgb_pred = rgb_pred.cpu()
    rgb_target = rgb_target.cpu()

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        # Input L*
        axes[i, 0].imshow(L_noisy[i, 0], cmap='gray')
        axes[i, 0].set_title('Input: Noisy L* (Lightness)')
        axes[i, 0].axis('off')

        output_img = rgb_pred[i].permute(1, 2, 0).clamp(0, 1).numpy()
        axes[i, 1].imshow(output_img)
        axes[i, 1].set_title('Output: Predicted Colors')
        axes[i, 1].axis('off')

        target_img = rgb_target[i].permute(1, 2, 0).clamp(0, 1).numpy()
        axes[i, 2].imshow(target_img)
        axes[i, 2].set_title('Target: Ground Truth')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    config = {
        # Data
        'train_dir': '../data/train2017',
        'val_dir': '../data/val2017',
        'batch_size': 32,
        'image_size': 128,
        'noise_std': 0.1,
        'num_workers': 4,

        'in_channels': 1,
        'out_channels': 2,
        'latent_dim': 128,

        'chromatic_weight': 5.0,
        'beta': 0.001,
        'use_kl_annealing': True,
        'warmup_epochs': 10,

        'num_epochs': 50,
        'learning_rate': 0.0001,

        'checkpoint_dir': 'checkpoints_lab',
        'output_dir': 'outputs_lab',
        'save_every': 5,
        'plot_every': 5,
    }

    Path(config['checkpoint_dir']).mkdir(exist_ok=True)
    Path(config['output_dir']).mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "=" * 70)
    print("Loading Data")
    print("=" * 70)

    train_dataset = LABColorizeDataset(
        config['train_dir'],
        image_size=config['image_size'],
        noise_std=config['noise_std']
    )

    val_dataset = LABColorizeDataset(
        config['val_dir'],
        image_size=config['image_size'],
        noise_std=config['noise_std']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    print("\n" + "=" * 70)
    print("Creating Model")
    print("=" * 70)

    model = CVAE(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        latent_dim=config['latent_dim']
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    print("\n" + "=" * 70)
    print("Loss Function: L*a*b* Chromatic-Weighted MSE + KL")
    print("=" * 70)

    loss_fn = LABCVAELoss(
        chromatic_weight=config['chromatic_weight'],
        beta=config['beta']
    ).to(device)

    print(f"Chromatic weight: {config['chromatic_weight']}")
    print(f"Beta (KL weight): {config['beta']}")
    print(f"KL annealing: {config['use_kl_annealing']}")
    print(f"Warmup epochs: {config['warmup_epochs']}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    train_history = {'total': [], 'color': [], 'kl': []}
    val_history = {'total': [], 'color': [], 'kl': []}

    best_val_loss = float('inf')

    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 70)

        if config['use_kl_annealing'] and epoch <= config['warmup_epochs']:
            beta_annealing = epoch / config['warmup_epochs']
        else:
            beta_annealing = 1.0

        print(f"Current beta: {beta_annealing * config['beta']:.6f}")

        # Train
        train_total, train_color, train_kl = train_epoch(
            model, train_loader, loss_fn, optimizer, device, beta_annealing
        )

        val_total, val_color, val_kl = validate(
            model, val_loader, loss_fn, device
        )

        train_history['total'].append(train_total)
        train_history['color'].append(train_color)
        train_history['kl'].append(train_kl)

        val_history['total'].append(val_total)
        val_history['color'].append(val_color)
        val_history['kl'].append(val_kl)

        print(f"Epoch {epoch} Summary:")
        print(f"  Train - Total: {train_total:.6f}, Color: {train_color:.6f}, KL: {train_kl:.6f}")
        print(f"  Val   - Total: {val_total:.6f}, Color: {val_color:.6f}, KL: {val_kl:.6f}")

        if val_total < best_val_loss:
            best_val_loss = val_total
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total,
                'config': config
            }
            torch.save(checkpoint, f"{config['checkpoint_dir']}/best_model_lab.pth")
            print(f"Best model saved (val_loss: {val_total:.6f})")

        if epoch % config['save_every'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_total,
                'config': config
            }
            torch.save(checkpoint, f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch}.pth")
            print(f"Checkpoint saved: epoch {epoch}")

        # Plot curves
        if epoch % config['plot_every'] == 0 or epoch == 1:
            plot_path = f"{config['output_dir']}/curves_epoch_{epoch}.png"
            plot_training_curves(train_history, val_history, plot_path)
            print(f"Training curves saved: {plot_path}")

            # Visualize results
            vis_path = f"{config['output_dir']}/results_epoch_{epoch}.png"
            visualize_results(model, val_loader, device, vis_path)
            print(f"Results saved: {vis_path}")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved in: {config['checkpoint_dir']}")
    print(f"Results saved in: {config['output_dir']}")

if __name__ == "__main__":
    main()
