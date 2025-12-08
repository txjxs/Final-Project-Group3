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
    a_denorm = ab[:, 0:1] * 255.0 - 128.0
    b_denorm = ab[:, 1:2] * 255.0 - 128.0

    lab = torch.cat([L_denorm, a_denorm, b_denorm], dim=1)

    lab_np = lab.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 3)

    rgb_list = []
    for i in range(lab_np.shape[0]):
        rgb = color.lab2rgb(lab_np[i])
        rgb = np.clip(rgb, 0, 1)
        rgb_list.append(rgb)

    rgb_np = np.stack(rgb_list, axis=0)  # (B, H, W, 3)
    rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()

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

