import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from monai.transforms import (
    Compose, RandFlipd, ToTensord
)
import random

def get_transforms(args):
    train_transform = Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        ToTensord(keys=["image", "label"]),
    ])
    val_transform = Compose([
        ToTensord(keys=["image", "label"]),
    ])
    return train_transform, val_transform


from torch.utils.data import WeightedRandomSampler


def get_binary_labels(dataset):
    labels = []
    for i in range(len(dataset)):
        _, mask = dataset[i]
        if mask.dim() == 3:  # C×H×W
            has_foreground = (mask.sum() > 0).item()
        elif mask.dim() == 2:  # H×W
            has_foreground = (mask.sum() > 0).item()
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
        labels.append(int(has_foreground))
    return labels


def get_loader(args):
    train_transform, val_transform = get_transforms(args)
    dataset = ColonDataset(args.image_dir, args.label_dir, transform=train_transform, ext=args.data_type)
    print(args.image_dir)
    if args.sub_dataset:
        indices = random.sample(range(len(dataset)), 100)
        dataset = Subset(dataset, indices)

    total_samples = len(dataset)
    train_size = int(args.split * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 设置 val 的 transform
    val_dataset.dataset.transform = val_transform
    use_sampler = args.classify
    # ✅ 使用 WeightedRandomSampler
    if use_sampler:
        print("🔄 Using WeightedRandomSampler for balanced training")
        train_labels = get_binary_labels(train_dataset)

        class_counts = np.bincount(train_labels)
        weights = 1.0 / class_counts
        sample_weights = [weights[label] for label in train_labels]

        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers)
    else:
        print("🚀 Using shuffle")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader


def get_sub_loader(args):
    train_transform, val_transform = get_transforms(args)
    dataset = ColonDataset(args.image_dir, args.label_dir, transform=train_transform)

    indices = random.sample(range(len(dataset)), 100)
    subset = Subset(dataset, indices)

    train_size = int(0.8 * len(subset))
    val_size = len(subset) - train_size
    train_dataset, val_dataset = random_split(subset, [train_size, val_size])

    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader
