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
    def __init__(self):
        super(Denoising_Model, self).__init__()
        self.encoder = Noisy_Encoder()
        self.decoder = Noisy_Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class UNet(nn.Module):
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




