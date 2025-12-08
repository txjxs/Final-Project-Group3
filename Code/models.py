import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class UNetEncoder(nn.Module):
    """
    U-Net Encoder with skip connections
    Processes noisy grayscale → latent space
    """

    def __init__(self, in_channels=1, latent_dim=128):
        super(UNetEncoder, self).__init__()

        self.latent_dim = latent_dim

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.flatten_size = 512 * 8 * 8
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        e1 = self.enc1(x) 
        e2 = self.enc2(e1)  
        e3 = self.enc3(e2)  
        e4 = self.enc4(e3) 

        e4_flat = e4.view(e4.size(0), -1)
        mu = self.fc_mu(e4_flat)
        logvar = self.fc_logvar(e4_flat)

        skip_connections = [e1, e2, e3, e4]
        return mu, logvar, skip_connections


class UNetDecoder(nn.Module):
    """
    U-Net Decoder with skip connections
    Processes latent space → clean RGB
    """

    def __init__(self, latent_dim=128, out_channels=3):
        super(UNetDecoder, self).__init__()

        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 512 * 8 * 8)

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512 + 512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, z, skip_connections):
        e1, e2, e3, e4 = skip_connections

        x = self.fc(z)
        x = x.view(x.size(0), 512, 8, 8)

        x = torch.cat([x, e4], dim=1)
        x = self.dec4(x)

        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x)

        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)

        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)

        return x


class CVAE(nn.Module):
    """
    CVAE: Noisy Grayscale → Clean RGB
    Uses U-Net architecture with skip connections
    """

    def __init__(self, in_channels=1, out_channels=3, latent_dim=128):
        super(CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = UNetEncoder(in_channels, latent_dim)
        self.decoder = UNetDecoder(latent_dim, out_channels)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        logvar = torch.clamp(logvar, min=-10, max=10)

        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon

        return z

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Noisy grayscale input (B, 1, H, W)

        Returns:
            reconstruction: Clean RGB output (B, 3, H, W)
            mu: Latent mean (B, latent_dim)
            logvar: Latent log variance (B, latent_dim)
        """

        mu, logvar, skip_connections = self.encoder(x)

        z = self.reparameterize(mu, logvar)

        reconstruction = self.decoder(z, skip_connections)

        return reconstruction, mu, logvar

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)

        with torch.no_grad():
            skip_connections = [
                torch.zeros(num_samples, 64, 64, 64).to(device),
                torch.zeros(num_samples, 128, 32, 32).to(device),
                torch.zeros(num_samples, 256, 16, 16).to(device),
                torch.zeros(num_samples, 512, 8, 8).to(device)
            ]
            samples = self.decoder(z, skip_connections)

        return samples




class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.e1 = conv_block(1, 64)
        self.e2 = conv_block(64, 128)
        self.e3 = conv_block(128, 256)
        self.e4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.b = conv_block(512, 1024)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d1 = conv_block(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d2 = conv_block(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d3 = conv_block(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d4 = conv_block(128, 64)

        # Output
        self.out = nn.Conv2d(64, 2, kernel_size=1)
        self.tanh = nn.Tanh()
    def forward(self, x):
        # Encoder
        c1 = self.e1(x)
        p1 = self.pool(c1)

        c2 = self.e2(p1)
        p2 = self.pool(c2)

        c3 = self.e3(p2)
        p3 = self.pool(c3)

        c4 = self.e4(p3)
        p4 = self.pool(c4)

        # Bottleneck
        b = self.b(p4)

        # Decoder (with Skip Connections)
        u1 = self.up1(b)
        cat1 = torch.cat((u1, c4), dim=1)
        dec1 = self.d1(cat1)

        u2 = self.up2(dec1)
        cat2 = torch.cat((u2, c3), dim=1)
        dec2 = self.d2(cat2)

        u3 = self.up3(dec2)
        cat3 = torch.cat((u3, c2), dim=1)
        dec3 = self.d3(cat3)

        u4 = self.up4(dec3)
        cat4 = torch.cat((u4, c1), dim=1)
        dec4 = self.d4(cat4)

        return self.tanh(self.out(dec4))


class ResNetUNet(nn.Module):
    """
    ResNet-18 Encoder + U-Net Decoder.
    """

    def __init__(self, n_classes=2):
        super().__init__()

        # 1. Pre-trained ResNet18
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base_layers = list(base_model.children())

        # 2.Encoder Layers
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        # 3. Decoder Layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up1 = self.conv_block(512 + 256, 256)
        self.conv_up2 = self.conv_block(256 + 128, 128)
        self.conv_up3 = self.conv_block(128 + 64, 64)
        self.conv_up4 = self.conv_block(64 + 64, 64)

        # 4. Final Output
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(64, n_classes, 1)
        self.tanh = nn.Tanh()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        # --- ENCODER ---
        x = input.repeat(1, 3, 1, 1)

        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)  # Bottleneck

        # --- DECODER ---

        x = self.upsample(layer4)
        if x.shape != layer3.shape:
            x = F.interpolate(x, size=layer3.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        if x.shape != layer2.shape:
            x = F.interpolate(x, size=layer2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        if x.shape != layer1.shape:
            x = F.interpolate(x, size=layer1.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        if x.shape != layer0.shape:
            x = F.interpolate(x, size=layer0.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up4(x)

        x = self.final_upsample(x)

        out = self.conv_last(x)
        return self.tanh(out)


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
            # 1x1 conv mixes the channels
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


class DoubleConv(nn.Module):
    """
    Standard Heavy Double Convolution
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class HeavyUNet(nn.Module):
    def __init__(self):
        super(HeavyUNet, self).__init__()

        # Exact same shape as LightweightUNet, but using Standard Convs
        self.inc = DoubleConv(3, 32)

        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1_conv = DoubleConv(512 + 256, 256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2_conv = DoubleConv(256 + 128, 128)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3_conv = DoubleConv(128 + 64, 64)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4_conv = DoubleConv(64 + 32, 32)

        self.outc = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

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
        return 'HeavyUNet'

