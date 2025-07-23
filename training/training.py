import torch
from tqdm import tqdm
from training_utils import *
from evaluation import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置日志记录

# 训练过程
def run_training(model, optimizer, scheduler, train_loader, val_loader, args):
    # 设置日志目录和路径
    logdir = os.path.join(args.logdir, f"{args.encoder}")
    checkpoint_path = os.path.join(logdir, f"{args.encoder}.pth")

    logger = setup_logger(logdir)
    early_stopping = EarlyStopping(patience=16)

    best_val_dice_score = 0.0
    train_losses, val_losses, val_dice_scores = [], [], []
    best_model_state_dict = None

    for epoch in range(args.max_epochs):
        model.train()
        train_loss = 0.0
        logger.info(f"Epoch {epoch + 1}/{args.max_epochs} - Training")
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.max_epochs} - Training", dynamic_ncols=True)

        for batch_idx, (images, targets) in enumerate(train_loader_tqdm):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            targets = targets.float()

            loss = dice_loss_func(outputs, targets)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            train_loader_tqdm.set_postfix(loss=loss.item(), refresh=False)

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logger.info(f"Epoch [{epoch + 1}/{args.max_epochs}], Loss: {avg_train_loss:.4f}")

        # 验证集评估
        val_loss, val_dice_score = evaluate(model, val_loader)
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice_score)
        logger.info(f"Validation Loss: {val_loss:.4f}, Dice: {val_dice_score:.4f}")

        scheduler.step(val_dice_score)

        # 保存最佳模型
        if val_dice_score > best_val_dice_score:
            best_val_dice_score = val_dice_score
            best_model_state_dict = model.state_dict()
            logger.info(f"🔥 New Best Dice: {best_val_dice_score:.4f} (Epoch {epoch + 1})")

        if early_stopping(val_dice_score):
            logger.info(f"⏹️ Early stopping at epoch {epoch + 1}")
            break

        torch.cuda.empty_cache()

    torch.save(best_model_state_dict, checkpoint_path)
    logger.info(f"✅ Best model saved to: {checkpoint_path}")

    visualize_training_progress(train_losses, val_losses, val_dice_scores, logdir)
    logger.info("🏁 Training complete.")
    return best_val_dice_score


# 评估过程

