import os
import re
import subprocess

# 要遍历的 base_dir 列表
base_dirs = [
    "./database/step1_ch3",
    "./database/step1_ch5",
    "./database/step2_ch3"
]

decoder_fusion_values = [True, False]
attention_types = ['se', 'none']
attention_encoders = ['mit_b0', 'mit_b1']
encoders = ['efficient0', 'efficient1']

logdir_base = "/root/autodl-tmp/segment/outputs/"

for base_dir in base_dirs:
    # 提取 in_channels：从路径中找 ch 后的数字
    match = re.search(r'ch(\d+)', base_dir)
    in_channels = match.group(1) if match else '3'

    for decoder_fusion in decoder_fusion_values:
        if decoder_fusion:
            for attention_type in attention_types:
                for attention_encoder in attention_encoders:
                    logdir = os.path.join(
                        logdir_base,
                        os.path.basename(base_dir)
                    )
                    command = [
                        "python", "main_online.py",
                        "--base_dir", base_dir,
                        "--in_channels", in_channels,
                        "--logdir", logdir,
                        "--decoder_fusion", "True",
                        "--attention_type", attention_type,
                        "--attention_encoder", attention_encoder
                    ]
                    print("Running:", " ".join(command))
                    subprocess.run(command)
        else:
            for encoder in encoders:
                logdir = os.path.join(
                    logdir_base,
                    os.path.basename(base_dir)
                )
                command = [
                    "python", "main_online.py",
                    "--base_dir", base_dir,
                    "--in_channels", in_channels,
                    "--logdir", logdir,
                    "--decoder_fusion", "False",
                    "--encoder", encoder
                ]
                print("Running:", " ".join(command))
                subprocess.run(command)
