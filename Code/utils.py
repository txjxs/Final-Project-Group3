import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import os


def seed_everything(seed=42):
    """
    Locks all random seeds for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print(f" Seed set to {seed}")

def calculate_psnr(img1, img2):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR).
    img1, img2: Tensors [N, C, H, W] in range [0, 1]
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))



def add_noise(img_tensor, noise_type='gaussian', factor=0.5):
    noisy_img = img_tensor.clone() #helps keep the main immage in memory clean
    if noise_type == 'gaussian':
        noise = torch.randn_like(img_tensor) * factor
        noisy_img += noise

    elif noise_type == 'salt_pepper':
        prob = torch.rand_like(img_tensor)
        # Salt (White pixels) -> Set to 1.0
        # If factor is 0.1, we want top 5% pixels to be white
        noisy_img[prob < (factor / 2)] = 1.0

        # Pepper (Black pixels) -> Set to 0.0
        # If factor is 0.1, we want bottom 5% pixels to be black
        noisy_img[prob > 1 - (factor / 2)] = 0.0

    else:
        raise ValueError(f"Unknown noise_type: {noise_type}. Use 'gaussian' or 'salt_pepper'.")

    return torch.clamp(noisy_img, 0, 1)

def plot_denoising_result(original, noisy, reconstructed=None , n=4):
    original = original.detach().cpu()
    noisy = noisy.detach().cpu()

    if reconstructed is not None:
        reconstructed = reconstructed.detach().cpu()

    plt.figure(figsize=(15,6))
    for i in range(n):
        ax = plt.subplot(3 if reconstructed is not None else 2, n, i+1)
        plt.imshow(original[i].permute(1,2,0))
        plt.title("Original")
        plt.axis('off')

        ax = plt.subplot(3 if reconstructed is not None else 2, n, i+1+n)
        plt.imshow(noisy[i].permute(1,2,0))
        plt.title("Noisy Input")
        plt.axis("off")

        if reconstructed is not None:
            ax = plt.subplot(3, n, i + 1 + 2 * n)
            plt.imshow(reconstructed[i].permute(1, 2, 0))
            plt.title("Reconstructed")
            plt.axis("off")
    save_path = 'test.png'
    plt.savefig(save_path)

