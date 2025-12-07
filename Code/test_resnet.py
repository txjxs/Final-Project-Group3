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