import torch
import torch.nn as nn


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
        self.out = nn.Conv2d(64, 2, kernel_size=1)  # Output 2 channels (ab)
        self.tanh = nn.Tanh()  # Constrain output to [-1, 1]

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



