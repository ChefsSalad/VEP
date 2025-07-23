import os
import numpy as np
import nibabel as nib
import cv2
import tifffile
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def normalize_image(image, mean, std):
    mask = image != 0
    normalized_image = (image - mean) / std
    return normalized_image * mask

def crop_black_borders(image):
    rows = np.any(image > 0, axis=1)
    cols = np.any(image > 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return np.zeros((512, 512))
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return image[rmin:rmax + 1, cmin:cmax + 1]


def clip_image(image, min_value=-60, max_value=200):
    return np.clip(image, min_value, max_value)


def save_image_and_label(image, label, image_filename, label_filename, save_format="npz"):
    if save_format == "npy":
        np.save(image_filename.replace('.tiff', '.npy'), image.astype(np.float32))
        np.save(label_filename.replace('.tiff', '.npy'), label.astype(np.uint8))

    elif save_format == "npz":
        npz_filename = image_filename.replace('.tiff', '.npz')
        np.savez_compressed(npz_filename, image=image.astype(np.float32), label=label.astype(np.uint8))

    elif save_format == "png":
        cv2.imwrite(image_filename.replace('.tiff', '.png'), image.astype(np.uint16))
        cv2.imwrite(label_filename.replace('.tiff', '.png'), label.astype(np.uint8))

    else:  # tiff
        tifffile.imwrite(image_filename, image.astype(np.uint16))
        tifffile.imwrite(label_filename, label.astype(np.uint8))


def process_image_slice(
    image_data, label_data, z, z_slices,
    output_image_dir, output_label_dir, output_label_vis_dir,
    do_padding=True, skip_empty_label=False, step=1, num_channels=3
):
    image_slice = image_data[:, :, z]
    label_slice = label_data[:, :, z]

    if skip_empty_label and np.max(label_slice) == 0:
        return

    image_slice = clip_image(image_slice, -60, 200)
    #image_slice = cv2.resize(image_slice, (512, 512), interpolation=cv2.INTER_CUBIC)

    if do_padding:
        assert num_channels % 2 == 1, "num_channels 必须为奇数"
        half = num_channels // 2
        offsets = [i * step for i in range(-half, half + 1)]  # e.g., [-4, -2, 0, 2, 4] 

        slices = []
        for offset in offsets:
            zi = z + offset
            if 0 <= zi < z_slices:
                img = image_data[:, :, zi]
            else:
                return 
                #img = image_data[:, :, z]  # 超出边界填当前 slice
            img = clip_image(img, -60, 200)
            #img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            img=   cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            slices.append(img)
           
        combined_image = np.stack(slices, axis=-1)
        
        #combined_image = cv2.rotate(combined_image, cv2.ROTATE_90_CLOCKWISE)

        #label_slice = cv2.resize(label_slice, (512, 512), interpolation=cv2.INTER_NEAREST)
        label_slice = cv2.rotate(label_slice, cv2.ROTATE_90_CLOCKWISE)
 
   

    else:
        combined_image = cv2.rotate(image_slice, cv2.ROTATE_90_CLOCKWISE)
        #label_slice = cv2.resize(label_slice, (512, 512), interpolation=cv2.INTER_NEAREST)
        out_channels=1,
        activation=None,
        label_slice = cv2.rotate(label_slice, cv2.ROTATE_90_CLOCKWISE)

    image_filename = os.path.join(output_image_dir, f"image_{z:03d}.tiff")
    label_filename = os.path.join(output_label_dir, f"label_{z:03d}.tiff")
    label_vis_filename = os.path.join(output_label_vis_dir, f"label_vis_{z:03d}.tiff")
    save_image_and_label(combined_image, label_slice, image_filename, label_filename)
    tifffile.imwrite(label_vis_filename, (label_slice * 255).astype(np.uint8))


def process_and_save_case(
    image_path, label_path, output_base_dir,
    do_padding=True, skip_empty_label=False, step=1, num_channels=3
):
    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)

    image_data = image_nii.get_fdata()
    label_data = label_nii.get_fdata()

    sample_name = os.path.basename(image_path).replace('.nii.gz', '')

    output_image_dir = os.path.join(output_base_dir, 'images', sample_name)
    output_label_dir = os.path.join(output_base_dir, 'labels', sample_name)
    output_label_vis_dir = os.path.join(output_base_dir, 'labels_vis', sample_name)

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_label_vis_dir, exist_ok=True)

    z_slices = image_data.shape[2]

    with ThreadPoolExecutor() as executor:
        for z in range(z_slices):
            executor.submit(
                process_image_slice,
                image_data, label_data, z, z_slices,
                output_image_dir, output_label_dir, output_label_vis_dir,
                do_padding, skip_empty_label, step, num_channels
            )


def preprocess_all_cases(
    image_dir, label_dir, output_base_dir,
    is_train=True, do_padding=True, step=1, num_channels=3
):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.nii.gz')]

    image_files.sort()
    label_files.sort()

    print(f"\n🔧 Preprocessing {len(image_files)} cases in {'train' if is_train else 'val'} set...")

    for image_file, label_file in tqdm(zip(image_files, label_files), total=len(image_files)):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        process_and_save_case(
            image_path=image_path,
            label_path=label_path,
            output_base_dir=output_base_dir,
            do_padding=do_padding,
            skip_empty_label=is_train,
            step=step,
            num_channels=num_channels
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess .nii.gz to .tiff slices with multi-slice padding support")
    parser.add_argument("--do_padding", type=bool, default=False, help="Use multi-slice padding")
    parser.add_argument("--do_splitting", type=bool, default=False, help="Whether to do splitting or not")
    parser.add_argument("--save_format", type=str, default="npz", choices=["npz","tiff", "npy", "png"],
                    help="save format: tiff / npy / png")
    parser.add_argument("--step", type=int, default=0, help="Slice interval step (0=continuous)")
    parser.add_argument("--num_channels", type=int, default=3, help="Total number of channels (must be odd)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    assert args.num_channels % 2 == 1, "num_channels must be an odd number!"

    
    # 👇 自动命名输出路径
  
    if args.do_splitting:
        base_input_dir = './database/Task10_Colon_split'
        datasets = [
            {'name': 'train', 'is_train': True},
            {'name': 'val','is_train':True}
        ]
        base_output_dir = os.path.join(
        './database/3d_eval',
        f"step{args.step}_ch{args.num_channels}")
    
    else:
        base_input_dir = './database/Task10_Colon'
        datasets = [
            {'name': 'train', 'is_train': True}
        ]
        base_output_dir = os.path.join(
        './database',
        f"step{args.step}_ch{args.num_channels}")
    if args.step>0:
        args.do_padding=True
    for dataset in datasets:
        image_dir = os.path.join(base_input_dir, dataset['name'], 'images')
        label_dir = os.path.join(base_input_dir, dataset['name'], 'labels')
        output_dir = os.path.join(base_output_dir, dataset['name'])

        preprocess_all_cases(
            image_dir=image_dir,
            label_dir=label_dir,
            output_base_dir=output_dir,
            is_train=dataset['is_train'],
            do_padding=args.do_padding,
            step=args.step,
            num_channels=args.num_channels
        )

    print(f"\n✅ All preprocessing done! Saved to: {base_output_dir}")
