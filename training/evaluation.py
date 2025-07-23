import torch
import numpy as np
import matplotlib.pyplot as plt
from loss import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def evaluate(model, val_loader):
    model.eval()
    val_loss = 0.0
    dice_scores = []

    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)

            targets = targets.float()
            outputs = model(images)
            y_pred = torch.sigmoid(outputs)

            dice = dice_iou(targets.cpu(), y_pred.cpu())
            loss = dice_loss_func(y_pred, targets)

            dice_scores.append(dice.item())
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_dice_score = np.mean(dice_scores)
    return avg_val_loss, avg_dice_score


# 可视化训练进度
def visualize_training_progress(train_losses, val_losses, val_dice_scores, logdir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_dice_scores, label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(logdir, 'training_visualization.png'))