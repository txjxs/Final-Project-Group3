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

    @staticmethod
    def name():
        return 'Denoising_Model'

    def __init__(self):
        super(Denoising_Model, self).__init__()
        self.encoder = Noisy_Encoder()
        self.decoder = Noisy_Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class UNet(nn.Module):

    def name(self):
        return 'UNet'


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


class DSConv(nn.Module):
    """
    Depthwise Separable Convolution Block
    1. Depthwise: Spatial filtering (groups=in_ch)
    2. Pointwise: Channel mixing (kernel_size=1)
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            # --- Depthwise ---
            # Groups=in_ch makes it depthwise (one filter per channel)
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            # --- Pointwise ---
            # 1x1 conv mixes the channels together
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class DSConvDouble(nn.Module):
    """
    The lightweight equivalent of the 'DoubleConv' block.
    It stacks two DSConv blocks to learn complex features efficiently.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            DSConv(in_ch, out_ch),
            DSConv(out_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class LightweightUNet(nn.Module):
    def __init__(self):
        super(LightweightUNet, self).__init__()

        # --- ENCODER ---
        # Initial Block: Standard Conv -> DSConv
        # We use a standard conv first to mix the raw RGB channels properly
        self.inc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DSConv(32, 32)
        )

        # Downsampling path
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DSConvDouble(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DSConvDouble(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DSConvDouble(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DSConvDouble(256, 512))  # Bottleneck

        # --- DECODER ---
        # Upsampling path with skip connections
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1_conv = DSConvDouble(512 + 256, 256)  # 1024 from bottleneck + 512 from skip

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2_conv = DSConvDouble(256 + 128, 128)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3_conv = DSConvDouble(128 + 64, 64)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4_conv = DSConvDouble(64 + 32, 32)

        # Final Output Block (1x1 Conv to get 3 channels)
        self.outc = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()  # Forces output to [0, 1]
        )

    def forward(self, x):
        # Down
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Up (Concatenation happens here)
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up1_conv(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up2_conv(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up3_conv(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up4_conv(x)

        return self.outc(x)

    @staticmethod
    def name():
        return 'LightweightUNet'
