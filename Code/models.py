import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

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


