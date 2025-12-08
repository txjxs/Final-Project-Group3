"""
Simple CVAE with U-Net architecture
Input: Noisy grayscale (1 channel)
Output: Clean RGB (3 channels)
"""

import torch
import torch.nn as nn
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
        # Encode with skip connections
        e1 = self.enc1(x)  # (B, 64, 64, 64)
        e2 = self.enc2(e1)  # (B, 128, 32, 32)
        e3 = self.enc3(e2)  # (B, 256, 16, 16)
        e4 = self.enc4(e3)  # (B, 512, 8, 8)

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
            nn.Sigmoid()  # Output in [0, 1]
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
        # Clamp for numerical stability
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
        # Encode
        mu, logvar, skip_connections = self.encoder(x)

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Decode
        reconstruction = self.decoder(z, skip_connections)

        return reconstruction, mu, logvar

    def sample(self, num_samples, device):
        """
        Generate samples from prior N(0, 1)
        Note: No skip connections available when sampling
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)

        # Create dummy skip connections
        with torch.no_grad():
            skip_connections = [
                torch.zeros(num_samples, 64, 64, 64).to(device),
                torch.zeros(num_samples, 128, 32, 32).to(device),
                torch.zeros(num_samples, 256, 16, 16).to(device),
                torch.zeros(num_samples, 512, 8, 8).to(device)
            ]
            samples = self.decoder(z, skip_connections)

        return samples
