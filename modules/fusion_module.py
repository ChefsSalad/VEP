import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 0. Direct Concat Fusion ----------
class ConcatFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, att_feat, conv_feat):
        fused = torch.cat([att_feat, conv_feat], dim=1)
        return fused


# ---------- 1. Gated Fusion ----------
class GatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Conv2d(channels * 2, 2, 1)

    def forward(self, att_feat, conv_feat):
        fusion_input = torch.cat([att_feat, conv_feat], dim=1)
        gate_logits = self.gate(fusion_input)
        gates = F.softmax(gate_logits, dim=1)
        g_att, g_conv = gates[:, 0:1], gates[:, 1:2]
        fused = g_att * att_feat + g_conv * conv_feat
        return torch.cat([fused, att_feat, conv_feat], dim=1)

# ---------- 2. Mutual Branch Co-Attention Fusion ----------
class MutalBranchCoAttention(nn.Module):
    def __init__(self, channels, reduction=16,  use_residual=True):
        super().__init__()
    
        self.use_residual = use_residual

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc_v = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        self.fc_t = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

 
    def forward(self, V, T):
        B, C, H, W = V.size()

        V_pool = self.avg_pool(V).view(B, C)
        T_pool = self.avg_pool(T).view(B, C)

        V_weight = self.fc_t(T_pool).view(B, C, 1, 1)
        T_weight = self.fc_v(V_pool).view(B, C, 1, 1)

        V_att = V * V_weight
        T_att = T * T_weight

        if self.use_residual:
            V_out = V + V_att
            T_out = T + T_att
        else:
            V_out = V_att
            T_out = T_att

        fused = torch.cat([V_out, T_out], dim=1)

        return fused







class MutualAttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_v = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.output_proj = nn.Conv2d(2 * channels, channels, kernel_size=1)

    def forward(self, V, T):
        B, C, H, W = V.shape
        N = H * W

        # Flatten spatial dims
        V_flat = V.view(B, C, -1)  # [B, C, N]
        T_flat = T.view(B, C, -1)  # [B, C, N]

        # Cross attention: V attends to T, T attends to V
        attn_V = torch.bmm(V_flat.transpose(1, 2), T_flat)  # [B, N, N]
        attn_T = torch.bmm(T_flat.transpose(1, 2), V_flat)  # [B, N, N]

        # Normalize
        attn_V = self.softmax(attn_V)
        attn_T = self.softmax(attn_T)

        # Apply attention
        V_att = torch.bmm(T_flat, attn_V.transpose(1, 2))  # [B, C, N]
        T_att = torch.bmm(V_flat, attn_T.transpose(1, 2))  # [B, C, N]

        # Reshape back
        V_att = V_att.view(B, C, H, W)
        T_att = T_att.view(B, C, H, W)

        # Fuse with residual attention
        V_fused = V + V_att
        T_fused = T + T_att

        fused = torch.cat([V_fused, T_fused], dim=1)  # [B, 2C, H, W]
        return fused


# class MutualAttentionFusion(nn.Module):
#     def __init__(self, channels, attention_size=8):
#         super().__init__()
#         self.attention_size = attention_size  # Target size for attention-based downsampling

#     def forward(self, feat1, feat2):
#         B, C, H, W = feat1.shape

#         # 1. Downsampling the features to the attention size using adaptive average pooling
#         V = F.adaptive_avg_pool2d(feat1, (self.attention_size, self.attention_size))  # Shape: (B, C, attention_size, attention_size)
#         T = F.adaptive_avg_pool2d(feat2, (self.attention_size, self.attention_size))  # Shape: (B, C, attention_size, attention_size)

#         # 2. Compute attention: matrix multiplication between V and T
#         M1 = torch.matmul(V, T.transpose(-1, -2))  # (B, C, attention_size, attention_size)
#         M2 = torch.matmul(T, V.transpose(-1, -2))  # (B, C, attention_size, attention_size)

#         # 3. Apply softmax to get attention weights (across the spatial dimensions)
#         N1 = F.softmax(M1, dim=-1)  # Apply softmax over the last dimension (attention_size)
#         N2 = F.softmax(M2, dim=-1)  # Apply softmax over the last dimension (attention_size)

#         O1 = torch.matmul(N1, V)  # (B, C, attention_size, attention_size) 矩阵乘法
#         O2 = torch.matmul(N2, T)  # (B, C, attention_size, attention_size) 矩阵乘法
        
#         # 5. Final fused attention output
#         A1 = O1 * V  # Further element-wise multiplication to refine V
#         A2 = O2 * T  # Further element-wise multiplication to refine T

#         # Return the fused outputs (A1 and A2), which are the attended feature maps
#         return A1, A2


# ---------- 3. Mutual Attention Fusion (MMMU Inspired) ----------
# class MutualAttentionFusion(nn.Module):
#     def __init__(self, channels, attention_size=128):
#         super().__init__()
#         self.attention_size = attention_size
#         self.scale = 1.0 / (channels ** 0.5)
#         self.out_proj = nn.Conv2d(channels * 3, channels, 1)

#     def forward(self, feat1, feat2):
#         B, C, H, W = feat1.shape

#         feat1_ds = F.adaptive_avg_pool2d(feat1, (self.attention_size, self.attention_size))
#         feat2_ds = F.adaptive_avg_pool2d(feat2, (self.attention_size, self.attention_size))
    

#         HW = self.attention_size * self.attention_size

#         V = feat1_ds.view(B, C, HW)
#         T = feat2_ds.view(B, C, HW)
#         Vt = V.transpose(1, 2)
#         Tt = T.transpose(1, 2)

#         M1 = torch.bmm(Vt, T) * self.scale
#         M2 = torch.bmm(Tt, V) * self.scale

#         N1 = torch.softmax(M1, dim=-1)
#         N2 = torch.softmax(M2, dim=-1)

#         O1 = torch.bmm(N1, T.transpose(1, 2)).transpose(1, 2).view(B, C, self.attention_size, self.attention_size)
#         O2 = torch.bmm(N2, V.transpose(1, 2)).transpose(1, 2).view(B, C, self.attention_size, self.attention_size)

#         O1_up = F.interpolate(O1, size=(H, W), mode='bicubic', align_corners=False)
#         O2_up = F.interpolate(O2, size=(H, W), mode='bicubic', align_corners=False)

#         F1 = O1_up * feat1 + feat1
#         F2 = O2_up * feat2 + feat2

#         return torch.cat([F1, F2], dim=1)




# ---------- Fusion Module Wrapper ----------
class FusionModule(nn.Module):
    def __init__(self, fusion_module: str):
        super().__init__()
        self.fusion_module_type = fusion_module
        self.module = None
        self.initialized = False

    def lazy_init(self, in_channels, device):
        if self.fusion_module_type == "projection":
            self.module = ConcatFusion(in_channels)
        elif self.fusion_module_type == "gated":
            self.module = GatedFusion(in_channels)
        elif self.fusion_module_type == "co_attention":
            self.module = ChannelCoAttention(in_channels)
        elif self.fusion_module_type == "mutual_attention":
            self.module = MutualAttentionFusion(in_channels)
        else:
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_module_type}")
        
        self.module.to(device)
        self.initialized = True

    def forward(self, a, b):
        if not self.initialized:
            self.lazy_init(a.shape[1], a.device)
        return self.module(a, b)
