"""
FC-EF-Res Network Architecture for Semantic Change Detection
Strategy 4.2: Integrated CD and LCM with Sequential Training

Based on Figure 5 from the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two convolutions"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out


class Encoder(nn.Module):
    """Encoder with residual blocks"""
    
    def __init__(self, in_channels=3, base_channels=16):
        super(Encoder, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Encoder levels
        self.encoder1 = self._make_layer(base_channels, base_channels, 2)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.encoder2 = self._make_layer(base_channels, base_channels*2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.encoder3 = self._make_layer(base_channels*2, base_channels*4, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.encoder4 = self._make_layer(base_channels*4, base_channels*8, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._make_layer(base_channels*8, base_channels*16, 2)
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Encoder with skip connections
        e1 = self.encoder1(x)
        x = self.pool1(e1)
        
        e2 = self.encoder2(x)
        x = self.pool2(e2)
        
        e3 = self.encoder3(x)
        x = self.pool3(e3)
        
        e4 = self.encoder4(x)
        x = self.pool4(e4)
        
        bottleneck = self.bottleneck(x)
        
        return bottleneck, [e4, e3, e2, e1]


class Decoder(nn.Module):
    """Decoder with skip connections"""
    
    def __init__(self, num_classes, base_channels=16):
        super(Decoder, self).__init__()
        
        # Decoder levels
        self.upconv4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, 
                                          kernel_size=2, stride=2)
        self.decoder4 = self._make_layer(base_channels*16, base_channels*8, 2)
        
        self.upconv3 = nn.ConvTranspose2d(base_channels*8, base_channels*4,
                                          kernel_size=2, stride=2)
        self.decoder3 = self._make_layer(base_channels*8, base_channels*4, 2)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels*4, base_channels*2,
                                          kernel_size=2, stride=2)
        self.decoder2 = self._make_layer(base_channels*4, base_channels*2, 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_channels*2, base_channels,
                                          kernel_size=2, stride=2)
        self.decoder1 = self._make_layer(base_channels*2, base_channels, 2)
        
        # Final classification layer
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, bottleneck, skip_connections):
        # Unpack skip connections
        e4, e3, e2, e1 = skip_connections
        
        # Decoder with skip connections
        x = self.upconv4(bottleneck)
        x = torch.cat([x, e4], dim=1)
        x = self.decoder4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.decoder3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.decoder2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.decoder1(x)
        
        # Final classification
        out = self.final_conv(x)
        
        return out


class SemanticChangeDetectionNet(nn.Module):
    """
    Strategy 4.2: Integrated Change Detection and Land Cover Mapping
    
    Architecture:
    - Two Siamese encoders (shared weights) for land cover mapping
    - One change detection branch using LCM features
    - Sequential training: Phase 1 (LCM only), Phase 2 (CD only)
    """
    
    def __init__(self, num_lc_classes=6, base_channels=16):
        super(SemanticChangeDetectionNet, self).__init__()
        
        # Shared encoder for both images (Siamese)
        self.encoder_lcm = Encoder(in_channels=3, base_channels=base_channels)
        
        # Two decoders for land cover mapping
        self.decoder_lcm = Decoder(num_classes=num_lc_classes, 
                                   base_channels=base_channels)
        
        # Project concatenated bottlenecks back to expected decoder input size
        self.cd_proj = nn.Conv2d(base_channels*16*2, base_channels*16,
                         kernel_size=1, bias=False)

        # Change detection decoder
        self.decoder_cd = Decoder(num_classes=2,
                          base_channels=base_channels)
    
    def forward(self, img1, img2, return_all=True):
        """
        Args:
            img1: First image (B, 3, H, W)
            img2: Second image (B, 3, H, W)
            return_all: If True, return LCM outputs too
        
        Returns:
            If return_all:
                lcm1, lcm2, change_map
            Else:
                change_map
        """
        # Phase 1: Land Cover Mapping (shared encoder)
        bottleneck1, skip1 = self.encoder_lcm(img1)
        bottleneck2, skip2 = self.encoder_lcm(img2)
        
        lcm1 = self.decoder_lcm(bottleneck1, skip1)
        lcm2 = self.decoder_lcm(bottleneck2, skip2)
        
        # Phase 2: Change Detection (using LCM features)
        # Concatenate bottleneck features
        combined = torch.cat([bottleneck1, bottleneck2], dim=1)  # [B, 512, 16, 16]
        combined = self.cd_proj(combined)                         # [B, 256, 16, 16]

        # Decode using skip connections from img1's encoder
        change_map = self.decoder_cd(combined, skip1)
        
        if return_all:
            return lcm1, lcm2, change_map
        else:
            return change_map
    
    def forward_lcm_only(self, img):
        """Forward pass for land cover mapping only"""
        bottleneck, skip = self.encoder_lcm(img)
        lcm = self.decoder_lcm(bottleneck, skip)
        return lcm
    
    def freeze_lcm(self):
        """Freeze land cover mapping parameters (for Phase 2 training)"""
        for param in self.encoder_lcm.parameters():
            param.requires_grad = False
        for param in self.decoder_lcm.parameters():
            param.requires_grad = False
        print("✓ LCM parameters frozen")
    
    def unfreeze_lcm(self):
        """Unfreeze land cover mapping parameters"""
        for param in self.encoder_lcm.parameters():
            param.requires_grad = True
        for param in self.decoder_lcm.parameters():
            param.requires_grad = True
        print("✓ LCM parameters unfrozen")
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test the network
if __name__ == "__main__":
    print("="*60)
    print("TESTING NETWORK ARCHITECTURE")
    print("="*60)
    
    # Create model
    model = SemanticChangeDetectionNet(num_lc_classes=6, base_channels=16)
    
    # Create dummy input
    batch_size = 2
    img1 = torch.randn(batch_size, 3, 256, 256)
    img2 = torch.randn(batch_size, 3, 256, 256)
    
    print(f"\nInput shapes:")
    print(f"  Image 1: {img1.shape}")
    print(f"  Image 2: {img2.shape}")
    
    # Forward pass
    print("\nForward pass (all outputs)...")
    lcm1, lcm2, change_map = model(img1, img2, return_all=True)
    
    print(f"\nOutput shapes:")
    print(f"  LCM 1: {lcm1.shape} (6 classes)")
    print(f"  LCM 2: {lcm2.shape} (6 classes)")
    print(f"  Change: {change_map.shape} (2 classes)")
    
    # Test freezing
    print(f"\nTotal trainable params: {model.get_trainable_params():,}")
    
    print("\nFreezing LCM branch...")
    model.freeze_lcm()
    print(f"Trainable params after freeze: {model.get_trainable_params():,}")
    
    model.unfreeze_lcm()
    print(f"Trainable params after unfreeze: {model.get_trainable_params():,}")
    
    print("\n✓ Network architecture test passed!")