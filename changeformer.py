"""
ChangeFormer - Siamese Network for Change Detection
Encoder: EfficientNet-B0 (pretrained, available in all timm versions)
Decoder: MLP fusion of 4 hierarchical feature scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.proj(x)


class ChangeFormerDecoder(nn.Module):
    def __init__(self, embed_dims, decoder_dim=128, num_classes=2):
        super().__init__()

        self.proj0 = MLP(embed_dims[0] * 2, decoder_dim)
        self.proj1 = MLP(embed_dims[1] * 2, decoder_dim)
        self.proj2 = MLP(embed_dims[2] * 2, decoder_dim)
        self.proj3 = MLP(embed_dims[3] * 2, decoder_dim)

        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        self.predict = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, features1, features2, target_size):
        projs = [self.proj0, self.proj1, self.proj2, self.proj3]
        outs = []

        for i, proj in enumerate(projs):
            f = torch.cat([features1[i], features2[i]], dim=1)
            f = proj(f)
            f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            outs.append(f)

        x = torch.cat(outs, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return x


class ChangeFormer(nn.Module):
    """
    Siamese ChangeFormer with EfficientNet-B0 encoder.
    Shared weights for both time points (Siamese).
    """

    # EfficientNet-B0 output channels at 4 stages
    EMBED_DIMS = [24, 40, 112, 320]

    def __init__(self, num_classes=2, decoder_dim=128, pretrained=True):
        super().__init__()

        # Shared EfficientNet-B0 encoder
        self.encoder = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)   # 4 stages at different resolutions
        )

        self.decoder = ChangeFormerDecoder(
            embed_dims=self.EMBED_DIMS,
            decoder_dim=decoder_dim,
            num_classes=num_classes
        )

    def forward(self, img1, img2):
        H, W = img1.shape[2], img1.shape[3]
        features1 = self.encoder(img1)
        features2 = self.encoder(img2)
        return self.decoder(features1, features2, target_size=(H, W))

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ChangeFormer(pretrained=False)
    img1 = torch.randn(2, 3, 256, 256)
    img2 = torch.randn(2, 3, 256, 256)
    out = model(img1, img2)
    print(f"Output shape: {out.shape}")   # (2, 2, 256, 256)
    print(f"Parameters:   {model.get_trainable_params():,}")