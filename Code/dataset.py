import os
from PIL import Image
from torch.utils.data import Dataset

class COCODataset(Dataset):
    """
    Custom class to load and preprocess COCO dataset
    """
    def __init__(self,root_dir, transform=None):
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


