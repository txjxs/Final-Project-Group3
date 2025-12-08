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