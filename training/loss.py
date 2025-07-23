import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F



# ✅ 全局 DiceLoss 实例，避免每次 new
dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=False)

def dice_loss_func(pred, target):
    return dice_loss_fn(pred, target)


def dice_iou(y_true, y_pred, thr=0.5, epsilon=0.001):
    y_pred = y_pred > thr
    inter = (y_true * y_pred).sum(dim=(2, 3))
    den = y_true.sum(dim=(2, 3)) + y_pred.sum(dim=(2, 3))
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean()
    return dice


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.33, dice_weight=0.67, smooth=1e-5):
        super(DiceBCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()  # BCE loss
        self.dice_loss = smp.losses.DiceLoss(mode='binary', smooth=smooth)  # Dice loss
        self.bce_weight = bce_weight  # BCE损失的权重
        self.dice_weight = dice_weight  # Dice损失的权重

    def forward(self, y_pred, y_true):
        # BCE损失
        bce = self.bce_loss(y_pred, y_true)

        # Dice损失
        dice = self.dice_loss(y_pred.sigmoid(), y_true)  # y_pred.sigmoid() 用于概率输出

        # 结合损失
        loss = self.bce_weight * bce + self.dice_weight * dice
        return loss

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()