import torch

def count_concat_fusion(m, x, y):
    m.total_ops = torch.zeros(1)

def count_gated_fusion(m, x, y):
    B, C, H, W = x[0].shape
    ops = (C * 2) * 2 * H * W + C * 3 * C * H * W
    m.total_ops = torch.Tensor([ops])

def count_channel_co_attention(m, x, y):
    B, C, H, W = x[0].shape
    red = max(1, C // 16)
    ops = B * (C * red + red * C) * 2 + B * C * H * W * 2
    m.total_ops = torch.Tensor([ops])

def count_mutual_attention(m, x, y):
    B, C, H, W = x[0].shape
    N = H * W
    ops = B * (C * N * N * 2 + C * N * 4)
    m.total_ops = torch.Tensor([ops])

def count_se_block(m, x, y):
    B, C, H, W = x[0].shape
    mid = 6 * C
    ops = B * C * H * W + B * C * mid + B * mid * C + B * C * H * W * 3
    m.total_ops = torch.Tensor([ops])

def count_channel_attention(m, x, y):
    B, C, H, W = x[0].shape
    mid = 16
    ops = 2 * B * C * H * W + B * C * mid + B * mid * C + B * C * H * W * 3
    m.total_ops = torch.Tensor([ops])

from modules.fusion_module import ConcatFusion, GatedFusion, ChannelCoAttention, MutualAttentionFusion
from modules.sa_head import SEBlock, ChannelAttention  # 如果你把它们分到 attention_modules.py 里了

custom_ops = {
    ConcatFusion: count_concat_fusion,
    GatedFusion: count_gated_fusion,
    ChannelCoAttention: count_channel_co_attention,
    MutualAttentionFusion: count_mutual_attention,
    SEBlock: count_se_block,
    ChannelAttention: count_channel_attention,
}
