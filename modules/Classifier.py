import segmentation_models_pytorch as smp
import torch.nn as nn
import torch


class SMPBinaryClassifier(nn.Module):
    def __init__(self, encoder_name='resnet18', in_channels=3, pretrained=False):
        super(SMPBinaryClassifier, self).__init__()
        
        # segmentation model
        self.seg_model = smp.Unet(
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=1,
            encoder_weights='imagenet' if pretrained else None
        )
        
        # 分类头
        self.pool = nn.AdaptiveAvgPool2d(1)  # 输出: [B, 1, 1, 1]
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # [B, 1, 1, 1] → [B, 1]
            nn.Dropout(p=0.3),
            nn.Linear(1, 32),            # 可以改成更大通道 if 输出更大
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 1)             # 输出 logits
        )

    def forward(self, x):
        seg_out = self.seg_model(x)        # shape: [B, 1, H, W]
        pooled = self.pool(seg_out)        # shape: [B, 1, 1, 1]
        out = self.classifier(pooled)      # shape: [B, 1]
        return out

class EncoderClassifier(nn.Module):
    def __init__(self, encoder_name='resnet34', in_channels=3, pretrained=False):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            weights='imagenet' if pretrained else None
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoder.out_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        feats = self.encoder(x)[-1]     # 最后一个 feature map
        pooled = self.pool(feats)
        out = self.classifier(pooled)
        return out
