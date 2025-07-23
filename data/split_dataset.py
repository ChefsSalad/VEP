import os
import random
import shutil

# 设置随机种子，保证每次划分结果一致
random.seed(42)

# 源数据文件夹
source_image_dir = '/root/autodl-tmp/segment/database/Task10_Colon/imagesTr'
source_label_dir = '/root/autodl-tmp/segment/database/Task10_Colon/labelsTr'

# 目标保存数据的根目录，可以自行修改
target_root = '/root/autodl-tmp/segment/database/Task10_Colon_split'

# 构建目标目录（训练集与验证集）
train_image_dir = os.path.join(target_root, 'train', 'imagesTr')
train_label_dir = os.path.join(target_root, 'train', 'labelsTr')
val_image_dir   = os.path.join(target_root, 'val', 'imagesTr')
val_label_dir   = os.path.join(target_root, 'val', 'labelsTr')

# 创建目标目录（如果不存在则创建）
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 获取所有 .nii.gz 文件（这里假设图像和标签文件名一一对应）
image_files = sorted([f for f in os.listdir(source_image_dir) if f.endswith('.nii.gz')])
label_files = sorted([f for f in os.listdir(source_label_dir) if f.endswith('.nii.gz')])

# 确保图像与标签顺序一致
assert image_files == label_files, "图像文件和标签文件不匹配！"

# 将图像和标签打包在一起，便于后续划分
combined_list = list(zip(image_files, label_files))
random.shuffle(combined_list)  # 打乱顺序

# 根据8:2比例划分数据
split_index = int(0.8 * len(combined_list))
train_set = combined_list[:split_index]
val_set = combined_list[split_index:]

# 定义复制文件的辅助函数
def copy_files(file_pairs, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    for img_file, lbl_file in file_pairs:
        shutil.copy(os.path.join(src_img_dir, img_file), os.path.join(dst_img_dir, img_file))
        shutil.copy(os.path.join(src_lbl_dir, lbl_file), os.path.join(dst_lbl_dir, lbl_file))

# 将划分好的数据分别复制到新的目录中
copy_files(train_set, source_image_dir, source_label_dir, train_image_dir, train_label_dir)
copy_files(val_set, source_image_dir, source_label_dir, val_image_dir, val_label_dir)

print(f"训练集数量: {len(train_set)}")
print(f"验证集数量: {len(val_set)}")
print("数据划分与复制完成！")
