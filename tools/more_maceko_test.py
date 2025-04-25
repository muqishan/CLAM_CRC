import os
import pickle
import openslide
import numpy as np
import h5py
import time
import torch
from torchvision.transforms import ToTensor, ToPILImage
import h5py
import random
from torchvision.transforms import ToPILImage
to_pil = ToPILImage()

random.seed(22)
# --- 参数设置 ---
np.random.seed(22)
output_dir = 'tools/normalized_images_macenko_torch'  # 保存归一化后图像的目录
os.makedirs(output_dir, exist_ok=True)
tools_dir = 'tools'               # tools 目录，用于保存中间结果
os.makedirs(tools_dir, exist_ok=True)
# 注意：这里的 pkl_path 根据实际保存的路径调整，若保存时文件路径为 tools/macenko_normalizer_torch.pkl，则请注意路径拼接
pkl_path = os.path.join(tools_dir, 'macenko_normalizer_torch.pkl')  

# --- 加载预先保存的归一化模型 ---
normalizer = torch.load(pkl_path)

# 将模型移动到合适设备，并设为评估模式
device = torch.device("cuda:0")
normalizer = normalizer.to(device)
normalizer.eval()


h5_filse = 'RESULTS_DIRECTORY/uni/h5_files'

h5s = random.choices([os.path.join(h5_filse,i) for i in os.listdir(h5_filse)],k=10)
slide_root_path = '/datasets2/lizhiyong/colon/colon20x20240908'

num_patches = 20
patch_width = 224
patch_height = 224
for h5_path in h5s:
    slide_name = os.path.basename(h5_path).replace('.h5','')
    slide_path = os.path.join(slide_root_path,slide_name+'.svs')
    with h5py.File(h5_path, 'r') as file:
        tarPatcheList = file['coords'][:]

    
    selected_indices = np.random.choice(tarPatcheList.shape[0], size=num_patches, replace=False)
    selected_coords = tarPatcheList[selected_indices]

    slide = openslide.OpenSlide(slide_path)

    for i, coord in enumerate(selected_coords):
        x, y = int(coord[0]), int(coord[1])
        # 提取 patch
        patch = slide.read_region((x, y), 0, (patch_width, patch_height))
        patch = patch.convert("RGB")
        normalized_patch_path = os.path.join(output_dir, slide_name)
        os.makedirs(normalized_patch_path,exist_ok=True)

        # 保存原始 patch 到 output_dir（保存为 PNG 文件）
        original_patch_path = os.path.join(normalized_patch_path, f"original_patch_{i}.png")
        patch.save(original_patch_path)
        print(f"已保存原始 patch {i} 到 {original_patch_path}")
        
        # 将 patch 转换为 tensor（范围 [0, 1]），并添加 batch 维度，然后移动到设备上
        patch_tensor = ToTensor()(patch).unsqueeze(0).to(device)
        
        # 记录归一化开始时间
        start_time = time.time()
        # 使用归一化模型对 patch 进行归一化处理
        normalized_tensor = normalizer(patch_tensor,cache_keys=[slide_name])
        # 记录归一化结束时间
        end_time = time.time()
        print(end_time - start_time,'ms')
        
        # 将归一化结果转换为 PIL 图像（先去除 batch 维度，并移动到 CPU）
        normalized_tensor = normalized_tensor.squeeze(0).cpu()
        normalized_image = ToPILImage()(normalized_tensor)
        
        # 保存归一化后的 patch 到 output_dir（保存为 PNG 文件）
        
        normalized_image.save(os.path.join(normalized_patch_path,f"normalized_patch_{i}.png"))
        print(f"已保存归一化 patch {i} 到 {normalized_patch_path}")

slide.close()