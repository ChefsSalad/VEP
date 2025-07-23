import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels):
        super(SEBlock, self).__init__()
        mid_channels =  6*in_channels  # 防止为0
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
    
        return residual+x * y


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()

        mid_channels = 16

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out

        # 计算 sigmoid 输出
        sigmoid_out = self.sigmoid(out)
        #print(sigmoid_out)
        return residual + x * sigmoid_out

class SliceAttentionHead(nn.Module):
    def __init__(self, in_channels, slice_attention_head='none'):
        super(SliceAttentionHead, self).__init__()
       
        if in_channels == 1 or slice_attention_head == 'none':
         
            self.module = nn.Identity()
        elif slice_attention_head == 'se':
            self.module = SEBlock(in_channels)
        elif slice_attention_head == 'ca':
            self.module = ChannelAttention(in_channels)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        

    def forward(self, x):
        return self.module(x)