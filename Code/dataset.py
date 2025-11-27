import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils import preprocess_image  # Import the helper function


class ColorizationDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Basic transforms (Resize, Augmentation)
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256))
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])

        # 1. Open image
        img = Image.open(img_path).convert("RGB")

        # 2. Use our Utility function to "decolorize" and split channels
        # This returns normalized L (input) and ab (target) tensors
        L, ab = preprocess_image(img, self.transform)

        return L, ab


