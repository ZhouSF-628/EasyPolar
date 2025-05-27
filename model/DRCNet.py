import torch
import torch.nn as nn
import torch.nn.functional as F

class DRCNet(nn.Module):
    def __init__(self, img_channel=3, base_channels=64):
        super(DRCNet, self).__init__()
        
        # 特征提取
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channel * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 可学习 threshold MLP
        self.threshold_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 全局平均池化
            nn.Flatten(),
            nn.Linear(base_channels, base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels // 2, 1),
            nn.Sigmoid()  # 输出0-1之间
        )
        
        # mask预测头
        self.mask_head = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 残差修正器（ResBlocks）
        self.residual_corrector = nn.Sequential(
            ResBlock(base_channels),
            ResBlock(base_channels),
            ResBlock(base_channels),
            nn.Conv2d(base_channels, img_channel, kernel_size=3, padding=1)
        )
        
    def forward(self, M_unpol, L_pol0):
        """
        Args:
            M_unpol: (B, 1, H, W) 无偏振图
            L_pol0: (B, 1, H, W) 左视角0°偏振图（已warp到中间视角）
        Returns:
            I0M: (B, 1, H, W) 修正后的中间视角0°偏振图
        """
        # 计算 residual
        residual = L_pol0 - M_unpol  # (B, 3, H, W)

        # 拼接输入
        x = torch.cat([M_unpol, residual], dim=1)  # (B, 6, H, W)
        feat = self.encoder(x)  # 提取特征

        # ---------- 替换的mask逻辑 ----------
        with torch.no_grad():
            # 计算 residual 的 L1 范数（多通道图像按通道平均）
            residual_l1 = torch.mean(torch.abs(residual), dim=1, keepdim=True)  # (B,1,H,W)
            
            # 小于阈值的区域认为是 residual == 0，生成 mask（表示需要修正的区域）
            mask = (residual_l1 < 1e-4).float()  # 阈值可调

        # 残差修正
        corrected_feat = feat * mask  # 只修正mask区域的特征
        corrected_r = self.residual_corrector(corrected_feat)  # (B, 1, H, W)

        # 融合：mask内用修正残差，mask外保留原residual
        fused_r = mask * corrected_r + (1 - mask) * residual
        IM = M_unpol + fused_r

        return IM

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return residual + out


if __name__ == "__main__":
    # 测试模型
    model = DRCNet().to('cuda')
    M_unpol = torch.randn(2, 3, 1024, 1024).to('cuda')
    L_pol0 = torch.randn(2, 3, 1024, 1024).to('cuda')
    
    output = model(M_unpol, L_pol0)
    print(output.shape)  # 应该是 (1, 1, 256, 256)