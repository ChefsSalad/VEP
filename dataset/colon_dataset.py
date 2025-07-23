import os
import numpy as np
import torch
from monai.data import Dataset
import tifffile

import cv2


class ColonDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, ext='npz'):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.ext = ext

        self.image_files = sorted([
            os.path.join(root, file)
            for root, _, files in os.walk(image_dir)
            for file in files if file.endswith(self.ext)
        ])
        self.label_files = sorted([
            os.path.join(root, file)
            for root, _, files in os.walk(label_dir)
            for file in files if file.endswith(self.ext)
        ])

        print(f"[ColonDataset] Loaded {len(self.image_files)} samples with extensions: {self.ext}")
        if not self.ext == 'npz':
            assert len(self.image_files) == len(self.label_files), "Image-label count mismatch!"

    def load_file(self, path):
        # ext = os.path.splitext(path)[-1].lower()
        if self.ext == 'npz':
            data = np.load(path)
            return data['image'], data['label']
        elif self.ext == 'npy':
            return np.load(path)
        elif self.ext in ['tiff', 'tif']:
            return tifffile.imread(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Cannot load image: {path}")
            if img.ndim == 2:
                return img
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):

        if self.ext == "npz":
            image, label = self.load_file(self.image_files[idx])
        else:
            image = self.load_file(self.image_files[idx])
            label = self.load_file(self.label_files[idx])

        if image.ndim == 2:
            image = image[np.newaxis, :, :]
        else:
            image = image.transpose(2, 0, 1)

        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)
            image = sample["image"]
            label = sample["label"]
            image = image.to(torch.float32)
            label = label.to(torch.float32)
        else:
            image = torch.from_numpy(image).float()
            label = torch.from_numpy(label).float()

        return image, label


class CaseDataset(Dataset):
    def __init__(self, image_root, label_root, transform=None, ext='npz'):
        self.case_names = sorted(os.listdir(image_root))
        self.image_root = image_root
        self.label_root = label_root
        self.transform = transform
        self.ext = ext

    def load_file(self, path):
        if self.ext == 'npz':
            data = np.load(path)
            return data['image'], data['label']
        elif self.ext == 'npy':
            return np.load(path)
        elif self.ext in ['tiff', 'tif']:
            return tifffile.imread(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Cannot load image: {path}")
            if img.ndim == 2:
                return img
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, idx):
        case_name = self.case_names[idx]
        case_image_dir = os.path.join(self.image_root, case_name)
        case_label_dir = os.path.join(self.label_root, case_name)

        image_files = sorted([f for f in os.listdir(case_image_dir) if f.endswith(self.ext)])
        label_files = sorted([f for f in os.listdir(case_label_dir) if f.endswith(self.ext)])

        if self.ext != 'npz':
            assert len(image_files) == len(label_files), f"Mismatch in slices for {case_name}"

        image_slices = []
        label_slices = []
        if self.ext == "npz":
            for img_file in image_files:

                img_path = os.path.join(case_image_dir, img_file)
                img, lbl = self.load_file(img_path)
                if img.ndim == 2:
                    img = img[..., np.newaxis]
                img = img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
                image_slices.append(img)

                lbl = lbl[np.newaxis, ...]  # (H, W) -> (1, H, W)

                label_slices.append(lbl)

        else:
            for img_file, lbl_file in zip(image_files, label_files):
                img_path = os.path.join(case_image_dir, img_file)
                lbl_path = os.path.join(case_label_dir, lbl_file)
                img = self.load_file(img_path)
                lbl = self.load_file(lbl_path)

                if img.ndim == 2:
                    img = img[..., np.newaxis]
                img = img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
                image_slices.append(img)

                lbl = lbl[np.newaxis, ...]  # (H, W) -> (1, H, W)
                label_slices.append(lbl)

        image_volume = torch.from_numpy(np.stack(image_slices, axis=0)).float()  # (D, C, H, W)
        label_volume = torch.from_numpy(np.stack(label_slices, axis=0)).long()  # (D, 1, H, W)

        # Apply MONAI transform if provided
        if self.transform:
            data = self.transform({"image": image_volume, "label": label_volume})
            image_volume = data["image"]
            label_volume = data["label"]

        return image_volume, label_volume.squeeze(1), case_name


