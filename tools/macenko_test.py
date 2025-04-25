import os
import pickle
import openslide
import numpy as np
import h5py
import time
import torch
from torchvision.transforms import ToTensor, ToPILImage
import h5py

from torchvision.transforms import ToPILImage
to_pil = ToPILImage()


# --- 参数设置 ---
np.random.seed(22)
output_dir = 'tools/normalized_images_macenko_torch'  
os.makedirs(output_dir, exist_ok=True)
tools_dir = 'tools'               
os.makedirs(tools_dir, exist_ok=True)

pkl_path = os.path.join(tools_dir, 'macenko_normalizer_torch.pkl')  

# --- 加载预先保存的归一化模型 ---
normalizer = torch.load(pkl_path)

# 将模型移动到合适设备，并设为评估模式
device = torch.device("cuda:0")
normalizer = normalizer.to(device)
normalizer.eval()

# --- 读取 tarPatcheList 坐标 ---
n1 = 'TCGA-AA-3529-01Z-00-DX1.99453fef-afe8-4a43-a64f-df2d48ef9e55_20x'
n2 = 'TCGA-A6-2677-01Z-00-DX1.dc0903dc-fef2-47ca-8f04-1ef25a4d8338_20x'
name = n1



h5file = f'RESULTS_DIRECTORY/uni/h5_files/{name}.h5'  
with h5py.File(h5file, 'r') as file:
    tarPatcheList = file['coords'][:]
print("tarPatcheList shape:", tarPatcheList.shape)

# --- 随机选择20个坐标 ---
num_patches = 10
selected_indices = np.random.choice(tarPatcheList.shape[0], size=num_patches, replace=False)
selected_coords = tarPatcheList[selected_indices]
print("随机选取的20个坐标:\n", selected_coords)

# --- 定义 patch size ---
patch_width = 224
patch_height = 224

# --- 打开目标 svs 切片 ---
target_path = f'/datasets2/lizhiyong/colon/colon20x20240908/{name}.svs'
print("使用的目标切片:", target_path)
slide = openslide.OpenSlide(target_path)

# --- 初始化记录归一化耗时的列表 ---
processing_times = []

# --- 遍历选定的20个坐标 ---
for i, coord in enumerate(selected_coords):
    x, y = int(coord[0]), int(coord[1])
    # 提取 patch
    patch = slide.read_region((x, y), 0, (patch_width, patch_height))
    patch = patch.convert("RGB")
    
    # 保存原始 patch 到 output_dir（保存为 PNG 文件）
    original_patch_path = os.path.join(output_dir, f"original_patch_{i}.png")
    patch.save(original_patch_path)
    print(f"已保存原始 patch {i} 到 {original_patch_path}")
    
    # 将 patch 转换为 tensor（范围 [0, 1]），并添加 batch 维度，然后移动到设备上
    patch_tensor = ToTensor()(patch).unsqueeze(0).to(device)
    
    # 记录归一化开始时间
    start_time = time.time()
    # 使用归一化模型对 patch 进行归一化处理
    normalized_tensor = normalizer(patch_tensor)
    # 记录归一化结束时间
    end_time = time.time()
    elapsed = end_time - start_time
    processing_times.append(elapsed)
    
    # 将归一化结果转换为 PIL 图像（先去除 batch 维度，并移动到 CPU）
    normalized_tensor = normalized_tensor.squeeze(0).cpu()
    normalized_image = ToPILImage()(normalized_tensor)
    
    # 保存归一化后的 patch 到 output_dir（保存为 PNG 文件）
    normalized_patch_path = os.path.join(output_dir, f"normalized_patch_{i}.png")
    normalized_image.save(normalized_patch_path)
    print(f"已保存归一化 patch {i} 到 {normalized_patch_path}")

slide.close()

# --- 输出耗时统计 ---
total_time = sum(processing_times)
average_time = total_time / len(processing_times) if processing_times else 0
print(f"20张 patch 总归一化耗时: {total_time:.4f} 秒, 平均耗时: {average_time:.4f} 秒")
