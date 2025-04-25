import os
import openslide
import numpy as np
import h5py
import torch
from torchvision.transforms import ToTensor
from torch_staintools.normalizer import NormalizerBuilder
import cv2
from PIL import Image

# --- 固定随机种子 ---
np.random.seed(19)
import os

cache_path = "tools/macenko_normalizer_torch.pkl"  # 你的缓存文件路径
if os.path.exists(cache_path):
    os.remove(cache_path)
    print(f"缓存文件 {cache_path} 已删除")

# --- 模板切片设置 ---
template_case = "TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F_20x"
print(f"使用模板切片: {template_case}")

# --- 路径设置 ---
slide_root = '/datasets2/lizhiyong/colon/colon20x20240908'
h5_root = 'RESULTS_DIRECTORY/uni/patches'

slide_path = os.path.join(slide_root, template_case + '.svs')
h5_path = os.path.join(h5_root, template_case + '.h5')

if not os.path.exists(slide_path):
    raise FileNotFoundError(f"未找到切片文件: {slide_path}")
if not os.path.exists(h5_path):
    raise FileNotFoundError(f"未找到 h5 文件: {h5_path}")

# --- 参数设置 ---
patch_width = 224
patch_height = 224

# 在整个流程中只打开一次 slide
slide = openslide.OpenSlide(slide_path)

# --- 定义函数：从 svs 文件中提取 patch ---
def extract_patch(x, y, width, height):
    patch = slide.read_region((x, y), 0, (width, height))
    patch = patch.convert("RGB")
    patch = np.array(patch)  # 转换为 numpy 数组
    return patch

# --- 定义函数：计算图像质量（使用拉普拉斯方差作为清晰度指标） ---
def calculate_quality(patch):
    # 将 RGB 图像转换为灰度图
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    # 计算拉普拉斯变换后的方差作为清晰度指标，值越大代表图像越清晰
    quality = cv2.Laplacian(gray, cv2.CV_64F).var()
    return quality

# --- 从 h5 文件中读取所有坐标 ---
with h5py.File(h5_path, 'r') as file:
    coords = file['coords'][:]  # 假设 shape 为 (N, 2)

# --- 遍历所有坐标，选择质量最好的 patch 作为模板 ---
best_quality = -1
best_patch = None
best_coord = None

for coord in coords:
    x, y = int(coord[0]), int(coord[1])
    try:
        patch = extract_patch(x, y, patch_width, patch_height)
        quality = calculate_quality(patch)
        print(f"坐标 {coord} 的质量得分: {quality:.2f}")
        if quality > best_quality:
            best_quality = quality
            best_patch = patch
            best_coord = coord
    except Exception as e:
        print(f"提取坐标 {coord} 时出错: {e}")

if best_patch is None:
    raise ValueError("没有提取到任何 patch！")

print(f"选择的最佳模板坐标为: {best_coord}，质量得分: {best_quality:.2f}")

# --- 保存最佳模板图片 ---
os.makedirs("tools", exist_ok=True)
save_path = os.path.join("tools", "best_template.png")
Image.fromarray(best_patch).save(save_path)
print(f"最佳模板图片已保存到: {save_path}")

# --- 使用选出的最佳 patch 作为模板 ---
template_image = best_patch
print("模板图像 shape:", template_image.shape)

# --- 转换为张量并移动到对应设备 ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
target_tensor = ToTensor()(template_image).unsqueeze(0).to(device)

# --- 构建并拟合归一化器 ---
normalizer = NormalizerBuilder.build('vahadane', use_cache=True, concentration_method='ista')
normalizer = normalizer.to(device)
normalizer.fit(target_tensor)

# --- 保存归一化模型 ---
output_pkl = 'tools/macenko_normalizer_torch1.pkl'
torch.save(normalizer, output_pkl)
print("归一化模型已保存到：", output_pkl)

# 最后关闭 slide 对象
slide.close()
