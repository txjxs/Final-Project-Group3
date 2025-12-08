import torch
import torch.nn as nn
import torch.nn.functional as F

class LABChromaticWeightedLoss(nn.Module):
    """
    MSE Loss weighted by color saturation in L*a*b* space
    In L*a*b*, saturation = sqrt(a^2 + b^2)
    - Gray: a=0, b=0, saturation=0
    - Vibrant: large |a| or |b|, saturation > 0
    """

    def __init__(self, chromatic_weight=3.0, base_weight=1.0):
        """
        Args:
            chromatic_weight: How much to emphasize saturated colors
            base_weight: Base weight for all regions
        """
        super(LABChromaticWeightedLoss, self).__init__()
        self.chromatic_weight = chromatic_weight
        self.base_weight = base_weight

    def compute_saturation(self, ab):
        """
        Compute color saturation from a*, b* channels

        Args:
            ab: (B, 2, H, W) - a*, b* channels normalized to [0, 1]

        Returns:
            saturation: (B, 1, H, W) - [0, 1] range
        """
        a = ab[:, 0:1] * 255.0 - 128.0
        b = ab[:, 1:2] * 255.0 - 128.0

        saturation = torch.sqrt(a**2 + b**2)

        saturation_norm = saturation / 180.0
        saturation_norm = torch.clamp(saturation_norm, 0, 1)

        return saturation_norm

    def forward(self, output, target):
        """
        Compute chromatic-weighted MSE on a*b* channels

        Args:
            output: (B, 2, H, W) predicted a*, b* in [0, 1]
            target: (B, 2, H, W) target a*, b* in [0, 1]

        Returns:
            weighted_mse: scalar loss
        """
        saturation = self.compute_saturation(target)

        weight = self.base_weight + self.chromatic_weight * saturation

        mse_per_pixel = (output - target) ** 2

        weighted_mse_per_pixel = mse_per_pixel * weight

        loss = weighted_mse_per_pixel.mean()

        return loss


class LABCVAELoss(nn.Module):
    """
    Complete CVAE loss for L*a*b* colorization

    Total loss = Chromatic-weighted MSE(a*b*) + beta * KL
    """

    def __init__(self, chromatic_weight=3.0, beta=0.001):
        super(LABCVAELoss, self).__init__()

        self.chromatic_loss = LABChromaticWeightedLoss(chromatic_weight=chromatic_weight)
        self.beta = beta

    def forward(self, output, target, mu, logvar):
        """
        Compute total CVAE loss for L*a*b*

        Args:
            output: (B, 2, H, W) predicted a*, b*
            target: (B, 2, H, W) target a*, b*
            mu: (B, latent_dim)
            logvar: (B, latent_dim)

        Returns:
            total_loss, color_loss, kl_loss
        """
        # Chromatic-weighted color loss
        color_loss = self.chromatic_loss(output, target)

        # KL divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_divergence)

        # Total loss
        total_loss = color_loss + self.beta * kl_loss

        return total_loss, color_loss, kl_loss


