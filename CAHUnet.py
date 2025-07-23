import torch.nn as nn
import segmentation_models_pytorch as smp
from utils.DataLoader import *
from modules.sa_head import SliceAttentionHead


class CAHUnet(nn.Module):
    def __init__(
        self,
        encoder,
        in_channels,
        out_channels,
        activation=None,
        head_attention_type='none',
        decoder_attention_type=None,
    ):
        super().__init__()

        if in_channels == 1:
            head_attention_type = 'none'

        self.slice_attention_head = SliceAttentionHead(in_channels, head_attention_type)

        self.seg_body = smp.Unet(
            encoder_name=encoder,  # 选择骨干网络，如 resnet34
            encoder_weights=None ,  # 预训练权重
            in_channels=in_channels,
            classes=out_channels,
            decoder_attention_type=decoder_attention_type,
            activation=activation,  # 训练时不使用激活，推理时再加
        )

    def forward(self, x):
        x = self.slice_attention_head(x)

        att_feats = self.attention_model.encoder(x)
        conv_feats = self.conv_model.encoder(x)

        att_out = self.attention_model.decoder(*att_feats)
        conv_out = self.conv_model.decoder(*conv_feats)

        fused = self.fusion(att_out, conv_out)

        masks = self.segmentation_head(fused)
        return masks