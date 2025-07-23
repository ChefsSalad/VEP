import torch.nn as nn
from training import initialization
from utils.modules import Activation

from modules.fusion_module import *
from modules.sa_head import *

class SegmentationHead(nn.Sequential):
    def __init__(self, out_channels, kernel_size=3, activation=None, upsampling=1):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation_type = activation
        self.upsampling = upsampling
        self.seg_head = None  # 延迟初始化的 Sequential 模块

    def forward(self, x):
        if self.seg_head is None:
            in_channels = x.shape[1]
            conv2d = nn.Conv2d(
                in_channels,
                self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            )
            up = (
                nn.UpsamplingBilinear2d(scale_factor=self.upsampling)
                if self.upsampling > 1
                else nn.Identity()
            )
            act = Activation(self.activation_type)

            self.seg_head = nn.Sequential(conv2d, up, act).to(x.device)

        return self.seg_head(x)      

# ---------- Integration in MultiModelFusion ----------
class MultiModelFusion(nn.Module):
    def __init__(
        self,
        attention_model,
        conv_model,
        in_channels,
        out_channels,
        activation=None,
        slice_attention_head='none',
        fusion_module='projection',
    ):
        super().__init__()

        if in_channels == 1:
            slice_attention_head = 'none'

        self.attention_model = attention_model
        self.conv_model = conv_model
        self.channel_attention = SliceAttentionHead(in_channels, slice_attention_head)

        self.fusion = FusionModule(fusion_module)

        self.segmentation_head = SegmentationHead(
            out_channels=out_channels,
            activation=activation,
        )

        initialization.initialize_head(self.segmentation_head)

    def forward(self, x):
        x = self.channel_attention(x)

        att_feats = self.attention_model.encoder(x)
        conv_feats = self.conv_model.encoder(x)

        att_out = self.attention_model.decoder(*att_feats)
        conv_out = self.conv_model.decoder(*conv_feats)

        fused = self.fusion(att_out, conv_out)

        masks = self.segmentation_head(fused)
        return masks