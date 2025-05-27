import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from model.DRCNet import DRCNet


# --- DoubleConv Block --- #
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,  kernel_size=3, norm='bn'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if norm == "bn":
            norm_fn = nn.BatchNorm2d
        elif norm == "in":
            norm_fn = nn.InstanceNorm2d
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1),
            norm_fn(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=1),
            norm_fn(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, norm='bn'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, norm=norm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, norm='bn'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm=norm)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# --- AttentionFusion Block --- #
class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fusion = torch.cat([x1, x2], dim=1)
        attention_map = self.attention(fusion)
        return x1 * attention_map + x2 * (1 - attention_map)

# --- Transformer Block --- #
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads=8, num_layers=6, feedforward_dim=2048):
        super(TransformerBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, feedforward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feedforward_dim, in_channels)
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x):
        # Attention layer
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)  # residual connection

        # Feedforward layer
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)  # residual connection

        return x
class Mlp(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(dim, dim * mult)
        self.fc2 = nn.Linear(dim * mult, dim)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super(Attention, self).__init__()
        self.heads = heads
        context_dim = context_dim or query_dim
        hidden_dim = max(query_dim, context_dim)
        # self.dim_head = int(hidden_dim / self.heads)
        self.dim_head = dim_head
        self.all_head_dim = self.heads * self.dim_head

        ## All linear layers (including query, key, and value layers and dense block layers) 
        ## preserve the dimensionality of their inputs and are tiled over input index dimensions # 
        # (i.e. applied as a 1 × 1 convolution).
        self.query = nn.Linear(query_dim, self.all_head_dim) # (b n d_q) -> (b n hd)
        self.key = nn.Linear(context_dim, self.all_head_dim) # (b m d_c) -> (b m hd)
        self.value = nn.Linear(context_dim, self.all_head_dim) # (b m d_c) -> (b m hd)
        self.out = nn.Linear(self.all_head_dim, query_dim) # (b n d) -> (b n d)
        # self.attn_dropout = nn.Dropout(dropout)
        # self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.heads, self.dim_head)
        x = x.view(*new_x_shape) # (b n hd) -> (b n h d)
        return x.permute(0, 2, 1, 3) # (b n h d) -> (b h n d)

    def forward(self, query, context=None):
        if context is None:
            context = query
        mixed_query_layer = self.query(query) # (b n d_q) -> (b n hd)
        mixed_key_layer = self.key(context) # (b m d_c) -> (b m hd)
        mixed_value_layer = self.value(context) # (b m d_c) -> (b m hd)

        query_layer = self.transpose_for_scores(mixed_query_layer) # (b h n d)
        key_layer = self.transpose_for_scores(mixed_key_layer) # (b h m d)
        value_layer = self.transpose_for_scores(mixed_value_layer) # (b h m d)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (b h n m)
        attention_scores = attention_scores / math.sqrt(self.dim_head) # (b h n m)
        attention_probs = self.softmax(attention_scores) # (b h n m)
        # attention_probs = self.attn_dropout(attention_probs) # (b h n m)

        context_layer = torch.matmul(attention_probs, value_layer) # (b h n m) , (b h m d) -> (b h n d)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # (b h n d)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        # attention_output = self.proj_dropout(attention_output)
        return attention_output


class Block(nn.Module):
    def __init__(self, hidden_size):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size)
        self.ffn_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size)
        

    def forward(self, x):
        x = self.attn(self.attention_norm(x)) + x
        x = self.ffn(self.ffn_norm(x)) + x
        return x
    
class ConfidenceNet(nn.Module):
    def __init__(self, in_channels, mid_channels=32):
        super(ConfidenceNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出范围为 [0, 1]
        )

    def forward(self, x):
        return self.net(x)

    
class TransUNet(nn.Module):
    def __init__(self, p_channels=3, i_channels=3, out_channels=3, dim=48, residual_num=8):
        super(TransUNet, self).__init__()

        # ConfidenceNet
        self.confidence_net = ConfidenceNet(in_channels=p_channels + i_channels)

        # Polar input network
        self.inc_p = DoubleConv(p_channels, dim)

        self.down1_p = Down(dim, dim * 2)
        self.down2_p = Down(dim * 2, dim * 4)
        self.down3_p = Down(dim * 4, dim * 8)
        self.down4_p = Down(dim * 8, dim * 8)

        # Intensity input network
        self.inc_i = DoubleConv(i_channels, dim)
        self.down1_i = Down(dim, dim)
        self.down2_i = Down(dim, dim * 2)
        self.down3_i = Down(dim * 2, dim * 4)

        # Fusion layers
        self.fusion4 = AttentionFusion(dim * 4)
        self.fusion3 = AttentionFusion(dim * 2)
        self.fusion2 = AttentionFusion(dim)
        self.fusion1 = AttentionFusion(dim)

        # Residual blocks
        self.resblock_layers = nn.ModuleList([
            Block(dim * 8) for _ in range(residual_num)
        ])

        # Upsampling layers
        self.up4 = Up(dim * 16, dim * 4)
        self.up3 = Up(dim * 8, dim * 2)
        self.up2 = Up(dim * 4, dim)
        self.up1 = Up(dim * 2, dim)

        # Output layer
        self.outc = OutConv(dim, out_channels)

    def forward(self, polar, inten):
        conf = self.confidence_net(torch.cat([polar, inten], dim=1))  # b, 1, h, w

        # Polar network forward
        xp1 = self.inc_p(polar)         # b, c, h, w
        xp1 = xp1 * conf            # b, c, h, w

        xp2 = self.down1_p(xp1)         # b, 2c, h/2, w/2
        xp3 = self.down2_p(xp2)         # b, 4c, h/4, w/4
        xp4 = self.down3_p(xp3)         # b, 8c, h/8, w/8
        xp5 = self.down4_p(xp4)         # b, 8c, h/16, w/16

        # Intensity network forward
        xi1 = self.inc_i(inten)         # b, c, h, w
        xi2 = self.down1_i(xi1)         # b, 2c, h/2, w/2
        xi3 = self.down2_i(xi2)         # b, 4c, h/4, w/4
        xi4 = self.down3_i(xi3)         # b, 8c, h/8, w/8

        # Residual blocks processing
        b, c, h, w = xp5.size()         
        x = xp5.view(b, c, -1).permute(0, 2, 1)     # b, h/8*w/8, 8c
        for resblock in self.resblock_layers:
            residual = resblock(x)
            x = residual
        x = x.permute(0, 2, 1).view(b, c, h, w)     # b, 8c, h/8, w/8

        # Upsampling and fusion
        # print("x shape", x.shape, "xp4.shape", xp4.shape)
        x = self.up4(x, xp4)            # b, 4c, h/8, w/8
        # print("x shape", x.shape, "xi4.shape", xi4.shape)
        x = self.fusion4(x, xi4)        # b, 4c, h/8, w/8

        # print("x shape", x.shape, "xp3.shape", xp3.shape)
        x = self.up3(x, xp3)            # b, 2c, h/4, w/4
        # print("x shape", x.shape, "xi3.shape", xi3.shape)
        x = self.fusion3(x, xi3)        # b, 2c, h/4, w/4

        # print("x shape", x.shape, "xp2.shape", xp2.shape)
        x = self.up2(x, xp2)            # b, c, h/2, w/2
        # print("x shape", x.shape, "xi2.shape", xi2.shape)
        x = self.fusion2(x, xi2)        # b, c, h/2, w/2

        # print("x shape", x.shape, "xp1.shape", xp1.shape)
        x = self.up1(x, xp1)            # up from 2c → c, h/2 → h
        # print("x shape", x.shape, "xi1.shape", xi1.shape)
        x = self.fusion1(x, xi1)        # fusion1 with xi1

        # Output layer
        logits = self.outc(x)
        return logits, conf

class TriplePolarNet(nn.Module):
    def __init__(self, p_channels=3, i_channels=3, out_channels=3):
        super(TriplePolarNet, self).__init__()

        self.warp0 = DRCNet(img_channel=i_channels)
        self.warp45 = DRCNet(img_channel=i_channels)

        self.aop_rec = TransUNet(p_channels=p_channels, i_channels=i_channels, out_channels=out_channels)
        # self.aop_rec.load_state_dict(torch.load('./result/train_v6_pre2/best.pth'))
        self.dolp_rec = TransUNet(p_channels=p_channels, i_channels=i_channels, out_channels=out_channels)

    def compute_polar(self, pol0, unpol, pol45):
        s0 = unpol * 2
        s1 = 2 * (pol0 - unpol)
        s2 = 2 * (pol45 - unpol)

        aop = torch.atan2(s2 + 1e-8, s1 + 1e-8) / 2
        dolp = torch.clamp(torch.sqrt(s1 ** 2 + s2 ** 2 + 1e-8) / (s0 + 1e-8), 0, 1)

        aop = (aop + torch.pi/2) / torch.pi

        return aop, dolp

    def forward(self, L_pol0, M_unpol, R_pol45):
        # Warping
        L_pol0 = self.warp0(M_unpol, L_pol0)
        R_pol45 = self.warp45(M_unpol, R_pol45)

        # Compute AoP and DoLP
        in_aop, in_dolp = self.compute_polar(L_pol0, M_unpol, R_pol45)

        # Reconstruction
        aop_out, aop_conf = self.aop_rec(in_aop, M_unpol)
        dolp_out, dolp_conf = self.dolp_rec(in_dolp, M_unpol)

        return L_pol0, R_pol45, aop_out, dolp_out, aop_conf, dolp_conf, in_aop, in_dolp

if __name__ == "__main__":
    # Test the model
    model = TriplePolarNet().to('cuda')

    b = 1
    h, w = 720, 720

    # Create dummy input tensors
    pol0 = torch.randn(b, 3, h, w).to('cuda')
    pol45 = torch.randn(b, 3, h, w).to('cuda')
    unpol = torch.randn(b, 3, h, w).to('cuda')

    # Forward pass
    L_pol0, R_pol45, aop_out, dolp_out = model(pol0, unpol, pol45)
    print(f"output shapes: L_pol0: {L_pol0.shape}, R_pol45: {R_pol45.shape}, aop_out: {aop_out.shape}, dolp_out: {dolp_out.shape}")