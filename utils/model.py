import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Conv Block =====
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ===== Squeeze-and-Excitation Fusion =====
class SEFusion(nn.Module):
    def __init__(self, ch1, ch2, fused_ch):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch1 + ch2, (ch1 + ch2) // 2),
            nn.ReLU(),
            nn.Linear((ch1 + ch2) // 2, ch1 + ch2),
            nn.Sigmoid()
        )
        self.fuse = nn.Conv2d(ch1 + ch2, fused_ch, 1)

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)  # [B, ch1+ch2, H, W]
        b, c, _, _ = x.shape
        y = self.global_pool(x).view(b, c)     # [B, ch]
        weights = self.fc(y).view(b, c, 1, 1)  # [B, ch, 1, 1]
        x = x * weights
        return self.fuse(x)

# ===== Attention Gate =====
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ===== Encoder =====
class Encoder(nn.Module):
    def __init__(self, in_ch, base_ch):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, base_ch)
        self.conv2 = ConvBlock(base_ch, base_ch * 2)
        self.conv3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip1 = self.conv1(x)
        x = self.pool(skip1)
        skip2 = self.conv2(x)
        x = self.pool(skip2)
        skip3 = self.conv3(x)
        x = self.pool(skip3)
        return [skip1, skip2, skip3], x

# ===== Decoder Block with Optional Attention =====
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.att = AttentionGate(F_g=out_ch, F_l=skip_ch, F_int=out_ch // 2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.att(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# ===== Final Dual-Encoder U-Net =====
class DualEncoderUNet(nn.Module):
    def __init__(self, in_ch1=1, in_ch2=1, out_ch=1, base_ch=32):
        super().__init__()
        self.encoder1 = Encoder(in_ch1, base_ch)  # boundary
        self.encoder2 = Encoder(in_ch2, base_ch)  # gradient

        # SE Fusion: (base_ch*4)*2 -> base_ch*8
        self.fusion = SEFusion(base_ch * 4, base_ch * 4, base_ch * 8)

        # Decoder with attention
        self.dec3 = DecoderBlock(base_ch * 8, base_ch * 4 * 2, base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4, base_ch * 2 * 2, base_ch * 2)
        self.dec1 = DecoderBlock(base_ch * 2, base_ch * 1 * 2, base_ch)

        self.final_conv = nn.Sequential(
            nn.Conv2d(base_ch, out_ch, 1),
            nn.Sigmoid()        
        )

    def forward(self, x1, x2):
        skips1, bottleneck1 = self.encoder1(x1)
        skips2, bottleneck2 = self.encoder2(x2)

        x = self.fusion(bottleneck1, bottleneck2)

        skip3 = torch.cat([skips1[2], skips2[2]], dim=1)
        x = self.dec3(x, skip3)

        skip2 = torch.cat([skips1[1], skips2[1]], dim=1)
        x = self.dec2(x, skip2)

        skip1 = torch.cat([skips1[0], skips2[0]], dim=1)
        x = self.dec1(x, skip1)

        return self.final_conv(x)
