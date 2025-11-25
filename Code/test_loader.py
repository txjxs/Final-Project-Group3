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



