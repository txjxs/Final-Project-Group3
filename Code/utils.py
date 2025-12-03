import numpy as np
from PIL import Image
from skimage import color
import torch
import matplotlib.pyplot as plt
import os


def preprocess_image(pil_image, transform=None):
    """
    The Master Decolorizer: Converts a PIL RGB image into L and ab tensors.
    Used by both the Dataset (training) and the App (inference).
    """
    # 1. Apply PyTorch transforms (Resize, Flip, etc.)
    if transform:
        pil_image = transform(pil_image)

    # 2. Convert to Numpy
    img_np = np.array(pil_image)

    # 3. Convert RGB to Lab
    # L range: [0, 100], a range: [-128, 127], b range: [-128, 127]
    img_lab = color.rgb2lab(img_np).astype("float32")

    # 4. Normalize to [-1, 1] range
    img_lab[:, :, 0] = (img_lab[:, :, 0] / 50.0) - 1.0  # L channel
    img_lab[:, :, 1:] = (img_lab[:, :, 1:] / 128.0)  # ab channels

    # 5. Convert to Tensor (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(img_lab.transpose((2, 0, 1)))

    # 6. Split into Input (L) and Target (ab)
    L = img_tensor[[0], ...]  # Shape: (1, 256, 256)
    ab = img_tensor[[1, 2], ...]  # Shape: (2, 256, 256)

    return L, ab


def lab_to_rgb(L_tensor, ab_tensor):
    """
    Converts a batch of L and ab tensors back to RGB numpy images.
    """
    rgb_imgs = []

    # Move to CPU and numpy
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

    # Visualize the first image in the batch
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
        print(f"💾 Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()


# --- WRAPPER FUNCTION FOR BACKWARD COMPATIBILITY ---
def save_results(L, ab_input, ab_output, epoch, folder="results"):
    """
    Wrapper for visualize_comparison to match the signature expected by train.py
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    path = f"{folder}/epoch_{epoch}.png"
    visualize_comparison(L, ab_input, ab_output, save_path=path)
