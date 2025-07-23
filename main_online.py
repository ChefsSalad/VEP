import torch.optim as optim
import numpy as np
import segmentation_models_pytorch as smp
from utils.training import run_training ,run_gan_training,run_classification_training # 复用训练逻辑
from encoder_hybrid import mixModel
from utils.DataLoader import *
from modules.critic import *
from modules.Classifier import *
from VEPUnet import *
import random


def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params / 1e6:.2f}M")  # 以百万为单位打印参数量
import csv

def save_results_to_csv(results, csv_path):
    # Define CSV header with the correct column order
    header = ['Visual Perception Method', 'step', 'channels', 'window', 'params', 'dice']
    
    # Open the file in append mode to add new rows without overwriting
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header only if the file is empty (check if the file exists)
        if file.tell() == 0:  # Check if file is empty
            writer.writerow(header)  # Write header
        
        for result in results:
            writer.writerow(result)  # Write each row


# 解析命令行参数
parser = argparse.ArgumentParser(description="SMP segmentation pipeline")

parser.add_argument("--roi_x", default=512, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=512, type=int, help="roi size in y direction")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="Learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="Optimizer")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="Weight decay")
parser.add_argument("--warmup_epochs", default=4, type=float, help="warmup epochs")
parser.add_argument("--warmup_start_lr", default=1e-4, type=float, help="max Learning rate")
parser.add_argument("--eta_min", default=1e-6, type=float, help="min Learning rate")

parser.add_argument("--sub_dataset", default=False, type=bool, help="use sub dataset")
parser.add_argument("--gan", default=False, type=bool, help="whether to use GAN")
parser.add_argument("--classify", default=False, type=bool, help="do classification")
parser.add_argument("--pretrained", default=False, type=bool, help="load weights")

parser.add_argument("--num_workers", default=8, type=int, help="Number of workers")
parser.add_argument('--activation', default=None, type=str, help="activation for last layer")
 #分割输出激活：'sigmoid', 'softmax', 'none'
parser.add_argument("--encoder_fusion", default=False, type=bool, help="Whether to fuse encoder")
parser.add_argument("--decoder_fusion", default=False, type=bool, help="Whether to fuse decoder")

parser.add_argument('--slice_attention_head', default='none', choices=['se', 'ca', 'none'], help="attemtion type")
parser.add_argument('--fusion_module', default='projection', choices=['projection', 'gated', 'co_attention','mutual_attention'],help=" whether to do gating")

parser.add_argument('--evaluate_3d', default=False, type=bool,help=" whether to do 3d evaluation")
parser.add_argument('--data_type', default='npz', choices=['tiff', 'npy', 'png', 'jpg'], help="data type")

parser.add_argument("--max_epochs", default=1024, type=int, help="Max training epochs")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--split", default=0.8, type=float, help="split size")


parser.add_argument("--in_channels", type=int, help="Number of input channels")
parser.add_argument("--out_channels", type=int, help="Number of output channels")
parser.add_argument("--gpu", default=0, type=int, help="GPU")


parser.add_argument("--base_dir", type=str,default="/root/autodl-tmp/segment/database", help="Which preprocessed step directory to use")

parser.add_argument("--step", type=int, help="Which preprocessed step directory to use")

parser.add_argument("--encoder", type=str, help="Encoder backbone")
parser.add_argument("--attention_encoder",  type=str, help="attention Encoder backbone")
parser.add_argument("--conv_encoder", type=str, help="conv Encoder backbone")

parser.add_argument("--model", default='Unet', type=str, help="Model architecture")

parser.add_argument("--attention_checkpoint", help="attention Start training from checkpoint")
parser.add_argument("--conv_checkpoint",  help="conv Start training from checkpoint")
parser.add_argument("--logdir", type=str, help="Directory for logs")

args = parser.parse_args()


def main():
    torch.backends.cudnn.benchmark = True  # 加速训练

    seed = 223  # 你可以选择任何你喜欢的种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 加速训练

    gpu=args.gpu
    torch.cuda.set_device(gpu)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)

    print(f"Using GPU: {gpu}")

    base_dir=args.base_dir

    args.image_dir = os.path.join(base_dir, "train", "images")
    args.label_dir = os.path.join(base_dir, "train", "labels")
    args.val_image_dir = os.path.join(base_dir, "val", "images")
    args.val_label_dir = os.path.join(base_dir, "val", "labels")
        
    
    print("📂Path：")
    print(f"  Train images: {args.image_dir}")
    print(f"  Train labels: {args.label_dir}")
    print(f"  Val images:   {args.val_image_dir}")
    print(f"  Val labels:   {args.val_label_dir}")

    # 1️⃣ 加载数据
    train_loader, val_loader = get_loader(args)
    if args.evaluate_3d:
        val_dataset = CaseDataset(args.val_image_dir, args.val_label_dir)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    if args.encoder_fusion:
        model = mixModel(args.attention_encoder, args.conv_encoder,in_channels=args.in_channels)
   
    elif args.decoder_fusion:
        attetion_model = smp.Unet(
            encoder_name=args.attention_encoder,  # 选择骨干网络，如 resnet34
            encoder_weights=None,  # 预训练权重
            in_channels=args.in_channels,
            classes=args.out_channels,
            #decoder_attention_type='scse',
            decoder_attention_type=None,
            activation=None,  # 训练时不使用激活，推理时再加
        )
        conv_model = smp.Unet(
            encoder_name=args.conv_encoder,  # 选择骨干网络，如 resnet34
            encoder_weights=None,  # 预训练权重
            in_channels=args.in_channels,
            classes=args.out_channels,
            #decoder_attention_type='scse',
            decoder_attention_type=None,
            activation=None,  # 训练时不使用激活，推理时再加
        )
        model = MultiModelFusion(
            attetion_model, 
            conv_model, 
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            slice_attention_head=args.slice_attention_head,
            activation=args.activation,
            fusion_module=args.fusion_module,
        )

    else:
        model = smp.Unet(
            encoder_name=args.encoder,  # 选择骨干网络，如 resnet34
            encoder_weights=None ,  # 预训练权重
            in_channels=args.in_channels,
            classes=args.out_channels,
            decoder_attention_type=None,
            activation=None,  # 训练时不使用激活，推理时再加
        )
        

    if args.classify:
        model = SMPBinaryClassifier(encoder_name=args.encoder, in_channels=args.in_channels,pretrained=args.pretrained)
        optimizer = optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
        # 5️⃣ 选择学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
        )
        run_classification_training(model, train_loader, train_loader,optimizer,scheduler,args)
        
    elif args.gan:
        NetS=model
        NetC=SegCritic(channel_dim=args.out_channels)
        optimizerS=optim.AdamW(NetS.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
        optimizerC=optim.AdamW(NetC.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizerS,
            mode='max',
            factor=0.5,
            patience=10,
        )
        run_gan_training(NetS, NetC, optimizerS, optimizerC, scheduler, train_loader, val_loader, args)

    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
        # 5️⃣ 选择学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
        )
        
        #print_model_params(model)
        print("in channel:",args.in_channels)
        model = model.cuda(gpu)
        dice=run_training(model, optimizer, scheduler, train_loader, val_loader, args)*100
        total_params = sum(p.numel() for p in model.parameters())

        # 计算Visual Perception Method
        if args.decoder_fusion:
            VPM = 'VEP'
            if args.fusion_module == 'co_attention':
                VPM += '-MBCA'
            if args.slice_attention_head == 'se':
                VPM += '-se'
        else:
            if args.encoder == 'mit_b2':
                VPM = 'Vit'
            elif args.encoder == 'efficientnet-b4':
                VPM = 'Conv'
            if args.slice_attention_head == 'se':
                VPM += '-se'
    
        # 计算window
        window = (args.in_channels - 1) / 2 * args.step
    
        # 保存结果
        results = [
            [
                VPM,
                args.step, 
                args.in_channels, 
                window,
                total_params / 1e6,  # 参数量以百万为单位
                dice,  # 假设dice是通过训练得到的，保存到结果中
            ]
        ]
        
        save_results_to_csv(results, 'training_results.csv')


if __name__ == "__main__":
    main()


