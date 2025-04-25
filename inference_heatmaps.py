# 接受一个路径，绘制该路径下所有图像的热图


'''

1.注意力热图，分辨率为level 2  概略图（5000左右分辨率）
2.与注意力热图同分辨率的 切片原始概略图
3.注意力分数最高/低的N张小图,切片原始小图，及热图小图颜色块儿
4.热图中第三项中小图的位置及排名，高为1-n，低为-1- -n（-1为最低)
5.切片原始概略图中第三项中小图的位置及排名，高为1-n，低为-1- -n（-1为最低)
6.第三项对应的小图，在免疫组化中的小图及位置

'''

import math
import os
import pandas as pd
import ast
from utils.eval_utils import initiate_model
from utils.utils import get_simple_loader
from dataset_modules.dataset_generic import Generic_MIL_Dataset
from models.model_clam import CLAM_SB, CLAM_MB
from models.model_mil import MIL_fc, MIL_fc_mc
import torch
import numpy as np
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import cv2

import openslide
from PIL import Image
import matplotlib.pyplot as plt

def colorize_slide(slidePath, NonCancerPos, CancerPos, CancerPosSource, heatmapSavePath,patch_size=(224,224), alpha=0.7, N=10):
    """
    对切片在 level2 层进行染色，并返回多个结果：
      1. 注意力热图（与底图融合后的热图）
      2. 切片原始概略图（level2 原始图）
      3. 在注意力热图上标注高/低注意力小图位置的图像（深红标注高，深蓝标注低）
      4. 在概略图上标注高/低注意力小图位置的图像（同上）
      5. 注意力最高的 N 张原始小图（从 level0 提取）
      6. 注意力最低的 N 张原始小图（从 level0 提取）
      7. 注意力最高的 N 张小图对应的热图（从最终融合的注意力 overlay 中提取）
      8. 注意力最低的 N 张小图对应的热图（从最终融合的注意力 overlay 中提取）

    参数：
        slidePath (str): 切片文件路径（包含 level0、level1、level2）
        NonCancerPos (numpy array): 非癌小图坐标数组，每行为 [pos_x, pos_y]（基于 level0 坐标）
        CancerPos (numpy array): 癌症小图坐标数组，每行为 [pos_x, pos_y]（基于 level0 坐标）
        CancerPosSource (numpy array or list): 癌症小图对应的注意力分数
        patch_size (tuple): 小图在 level0 下的尺寸（默认 224x224），后续按层级缩放
        alpha (float): 融合系数（0～1），表示热图与底图的混合比例
        N (int): 返回最高/最低注意力小图的数量，默认 10

    返回：
        (attention_heatmap, summary_img, annotated_heatmap_img, annotated_summary_img,
         high_patch_list, low_patch_list, high_heatmap_patch_list, low_heatmap_patch_list)
    """
    # 1. 读取切片并获取 level2 底图及缩放比例，同时获得原始概略图（summary_img）
    slide = openslide.OpenSlide(slidePath)
    level = 2
    level2_dim = slide.level_dimensions[level]   # (width, height)
    level0_dim = slide.level_dimensions[0]
    scale = np.array([level2_dim[0] / level0_dim[0], level2_dim[1] / level0_dim[1]])
    # summary_img直接为 level2 原始图
    summary_img = slide.read_region((0,0), level, level2_dim).convert("RGB")
    base_img = np.array(summary_img)
    patch_size_level = np.ceil(np.array(patch_size) * scale).astype(int)
    
    # 2. 坐标转换：将基于 level0 的坐标转换为 level2 坐标
    CancerPos = np.array(CancerPos)
    NonCancerPos = np.array(NonCancerPos)
    CancerPos_level = np.ceil(CancerPos * scale).astype(int)
    NonCancerPos_level = np.ceil(NonCancerPos * scale).astype(int)
    
    # 3. 归一化癌症注意力分数并设置 colormap
    CancerPosSource = np.array(CancerPosSource)
    aggregated_scores = CancerPosSource.flatten()
    if aggregated_scores.shape[0] != CancerPos.shape[0]:
        if aggregated_scores.shape[0] == 1:
            aggregated_scores = np.repeat(aggregated_scores, CancerPos.shape[0])
        else:
            raise ValueError(f"CancerPosSource 的数量与 CancerPos 坐标数量不一致！{slidePath}CancerPos:{CancerPos.shape}CancerPosSource:{aggregated_scores.shape}project:{heatmapSavePath}")
    if aggregated_scores.max() - aggregated_scores.min() > 0:
        norm_scores = (aggregated_scores - aggregated_scores.min()) / (aggregated_scores.max() - aggregated_scores.min())
    else:
        norm_scores = np.zeros_like(aggregated_scores)
    cmap = plt.get_cmap('coolwarm')
    
    # 4. 构建癌症区域连续注意力 overlay 与计数器
    overlay_cancer = np.zeros((level2_dim[1], level2_dim[0]), dtype=float)
    count_cancer = np.zeros((level2_dim[1], level2_dim[0]), dtype=np.uint16)
    for i, coord in enumerate(CancerPos_level):
        x, y = coord
        x_end = min(x + patch_size_level[0], level2_dim[0])
        y_end = min(y + patch_size_level[1], level2_dim[1])
        overlay_cancer[y:y_end, x:x_end] += norm_scores[i]
        count_cancer[y:y_end, x:x_end] += 1
    valid_mask = count_cancer > 0
    overlay_avg = np.zeros_like(overlay_cancer)
    overlay_avg[valid_mask] = overlay_cancer[valid_mask] / count_cancer[valid_mask]
    
    # 5. 对 overlay 进行高斯模糊平滑
    smooth_factor = 3
    kernel_size = np.maximum(1, (patch_size_level * smooth_factor)).astype(int)
    kernel_size = (kernel_size // 2) * 2 + 1
    overlay_smooth = cv2.GaussianBlur(overlay_avg, tuple(kernel_size), 0)
    
    # 6. 使用 colormap 将平滑后的 overlay 映射为彩色图（癌症区域热图）
    overlay_norm = np.clip(overlay_smooth, 0, 1)
    colored_cancer = (cmap(overlay_norm)[:, :, :3] * 255).astype(np.uint8)
    
    # 7. 构建非癌区域 mask 并平滑
    mask_non_cancer = np.zeros((level2_dim[1], level2_dim[0]), dtype=np.float32)
    for coord in NonCancerPos_level:
        x, y = coord
        x_end = min(x + patch_size_level[0], level2_dim[0])
        y_end = min(y + patch_size_level[1], level2_dim[1])
        mask_non_cancer[y:y_end, x:x_end] = 1.0
    kernel_mask = np.maximum(1, (patch_size_level // 2)).astype(int)
    kernel_mask = (kernel_mask // 2) * 2 + 1
    mask_non_smooth = cv2.GaussianBlur(mask_non_cancer, tuple(kernel_mask), 0)
    mask_non_smooth = np.clip(mask_non_smooth, 0, 1)
    
    # 8. 生成非癌区域彩色图（固定为灰色）
    non_cancer_color = np.array([199, 242, 180], dtype=np.uint8)
    non_cancer_overlay = np.ones_like(base_img) * non_cancer_color
    
    # 9. 融合癌症区域与非癌区域
    final_overlay = (colored_cancer.astype(np.float32) * (1 - mask_non_smooth[..., None]) +
                     non_cancer_overlay.astype(np.float32) * mask_non_smooth[..., None]).astype(np.uint8)
    mask_cancer_bin = (count_cancer > 0).astype(np.uint8)
    mask_non_bin = (mask_non_cancer > 0).astype(np.uint8)
    total_mask = np.clip(mask_cancer_bin + mask_non_bin, 0, 1)
    final_overlay[total_mask == 0] = [255, 255, 255]
    
    # 10. 与底图混合，生成最终注意力热图
    blended = cv2.addWeighted(base_img, 1 - alpha, final_overlay, alpha, 0)
    attention_heatmap = Image.fromarray(blended)
    
    # -------------------------------
    # 11. 提取注意力最高/最低的 N 张小图（原始小图）并在注意力热图与概略图上标注位置
    # 同时增加提取对应的热图小图（从 final_overlay 中提取）
    # -------------------------------
    indices = np.arange(len(norm_scores))
    sorted_high = indices[np.argsort(-norm_scores)]
    sorted_low = indices[np.argsort(norm_scores)]
    high_patch_list = []
    low_patch_list = []
    high_heatmap_patch_list = []
    low_heatmap_patch_list = []
    
    # 复制一份用于标注（注意：直接在 attention_heatmap 和 summary_img 上绘制标注）
    annotated_heatmap = np.array(attention_heatmap.copy())
    annotated_summary = np.array(summary_img.copy())
    # 设置标注参数：高注意力使用深红色，低注意力使用深蓝色
    high_color = (139, 0, 0)  # 深红色
    low_color  = (0, 0, 139)  # 深蓝色
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    
    # 对于高注意力 patch
    for rank, idx in enumerate(sorted_high[:N], start=1):
        # 提取原始小图（level0 下）
        coord0 = tuple(CancerPos[idx])
        patch_img = slide.read_region(coord0, 0, patch_size).convert("RGB")
        high_patch_list.append(patch_img)
        # 提取对应的热图小图（从 final_overlay 中）
        coord_level = CancerPos_level[idx]
        x, y = coord_level
        x_end = min(x + patch_size_level[0], level2_dim[0])
        y_end = min(y + patch_size_level[1], level2_dim[1])
        patch_heatmap = final_overlay[y:y_end, x:x_end]
        high_heatmap_patch_list.append(Image.fromarray(patch_heatmap))
        # 在标注图上绘制边框及序号（正序，1表示最高）
        cv2.rectangle(annotated_heatmap, (x, y), (x_end, y_end), high_color, thickness)
        cv2.putText(annotated_heatmap, str(rank), (x, y - 5), font, font_scale, high_color, font_thickness)
        cv2.rectangle(annotated_summary, (x, y), (x_end, y_end), high_color, thickness)
        cv2.putText(annotated_summary, str(rank), (x, y - 5), font, font_scale, high_color, font_thickness)
    
    # 对于低注意力 patch（标注为负数，-1 表示最低）
    for rank, idx in enumerate(sorted_low[:N], start=1):
        patch_img = slide.read_region(tuple(CancerPos[idx]), 0, patch_size).convert("RGB")
        low_patch_list.append(patch_img)
        coord_level = CancerPos_level[idx]
        x, y = coord_level
        x_end = min(x + patch_size_level[0], level2_dim[0])
        y_end = min(y + patch_size_level[1], level2_dim[1])
        patch_heatmap = final_overlay[y:y_end, x:x_end]
        low_heatmap_patch_list.append(Image.fromarray(patch_heatmap))
        cv2.rectangle(annotated_heatmap, (x, y), (x_end, y_end), low_color, thickness)
        cv2.putText(annotated_heatmap, "-" + str(rank), (x, y - 5), font, font_scale, low_color, font_thickness)
        cv2.rectangle(annotated_summary, (x, y), (x_end, y_end), low_color, thickness)
        cv2.putText(annotated_summary, "-" + str(rank), (x, y - 5), font, font_scale, low_color, font_thickness)
    
    annotated_heatmap_img = Image.fromarray(annotated_heatmap)
    annotated_summary_img = Image.fromarray(annotated_summary)
    
    # -------------------------------
    # 12. 返回结果
    # -------------------------------
    return (attention_heatmap, summary_img, annotated_heatmap_img, annotated_summary_img,
            high_patch_list, low_patch_list, high_heatmap_patch_list, low_heatmap_patch_list)





class PTFDataset(Dataset):
    def __init__(self, file_paths):
        """
        初始化数据集，接受一个包含 pt 文件路径的列表。
        """
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        返回特征向量矩阵和文件名。
        """
        file_path = self.file_paths[idx]
        data = torch.load(file_path)  # 加载 pt 文件
        siteInfo = pd.read_csv(file_path.replace('pt_files','predicts').replace('.pt','.h5.csv'))
        return data, os.path.basename(file_path), siteInfo

def collate_fn_with_filename(batch):
    """
    自定义 collate_fn，处理特征矩阵和文件名。
    """
    features = torch.cat([item[0] for item in batch], dim = 0)
    filenames = [item[1] for item in batch][0]
    siteInfo = [item[2] for item in batch][0]
    return features, filenames,siteInfo

def get_simple_loader(dataset, batch_size=1, num_workers=1):
    """
    返回一个数据加载器，用于加载特征向量矩阵，并同时返回文件名。
    """
    kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers}
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=SequentialSampler(dataset), 
        collate_fn=collate_fn_with_filename, 
        **kwargs
    )
    return loader

def initModelArgs(args_path,featureModel):
    # args_path = path.replace(os.path.basename(path),'experiment_task_1_tumor_vs_normal.txt')
    # print('args_path',args_path)
    with open(args_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
        data_dict = ast.literal_eval(file_content)
    data_dict['embed_dim'] = EMBED_DIM[featureModel]
    return data_dict

def loadModel(model_args,modelPath):
    print('Init Model')    
    model_dict = {"dropout": model_args['use_drop_out'], 'n_classes': 2, "embed_dim": model_args['embed_dim']}
    
    # if model_args['model_size'] is not None and model_args['model_type'] in ['clam_sb', 'clam_mb']:
    #     model_dict.update({"size_arg": model_args['model_type']})
    # model_size =  model_args.get['model_size']
    # k_sample
    model_dict.update({"k_sample": model_args.get('B')})
    model_dict.update({"size_arg": model_args.get('model_size')})
    model_dict.update({"dropout": model_args.get('use_drop_out')})
    if model_args['model_type'] =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif model_args['model_type'] =='clam_mb':
        model = CLAM_MB(**model_dict)
    else: # args.model_type == 'mil'
        model_dict.pop('k_sample')
        model = MIL_fc(**model_dict)

    ckpt = torch.load(modelPath)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


BASE_PATH = 'best_model_one'
DEVICE = 'cuda:0'
# MULTI_INSTANCE_MODELS = ['mil']
MULTI_INSTANCE_MODELS = ['clam','mil']
# PROJECT_NAMES = ['Autophagy','Stemness']
PROJECT_NAMES = ['Angiogenesis', 'Autophagy','Stemness']
# PROJECT_NAMES = ['MSI', 'TMB']
# PROJECT_NAMES = ['Acetylation'] 
FEATURE_MODELS = ['CTrans','uni','resnet']
EXTERNAL_DATA_PATH = 'ExternalDataCrc/wsi/20250327/RESULTS_DIRECTORY/***/pt_files' #
EXTERNAL_WSI_PATH = 'ExternalDataCrc/wsi/20250327/split'  # 对应的切片路径，获取缩略图
EMBED_DIM = {
                'uni': 1024,
                'CTrans':768,
                'resnet':1024    
            }
HeatmapSavePath = 'ExternalDataCrc/wsi/20250327/heatmaps'
for multi_instance in MULTI_INSTANCE_MODELS:  # method: mil/clam
    base_multi_instance_path = os.path.join(BASE_PATH, multi_instance)
    for project in PROJECT_NAMES:  # process: Angiogenesis/Autophagy/Stemness
        project_path = os.path.join(base_multi_instance_path, project)
        for featurn in FEATURE_MODELS:  # model: uni/resnet/CTrans
            modelRootPath = os.path.join(os.path.join(project_path, featurn),os.listdir(os.path.join(project_path, featurn))[0])
            modelConfig = initModelArgs(os.path.join(modelRootPath,'experiment_task_1_tumor_vs_normal.txt'),featurn)
            model = loadModel(modelConfig, os.path.join(modelRootPath,[i for i in os.listdir(modelRootPath) if i.endswith('.pt')][0]))

            ExternalDataPath = EXTERNAL_DATA_PATH.replace('***',featurn)
            ExternalDataPath = [os.path.join(ExternalDataPath,i) for i in os.listdir(ExternalDataPath)]
            dataset = PTFDataset(ExternalDataPath)

            # 获取数据加载器
            dataLoader = get_simple_loader(dataset, batch_size=1, num_workers=1)

            for batch_idx, (data, svsName,siteInfo) in enumerate(dataLoader):
                data = data.to(DEVICE)
                with torch.no_grad():
                    logits, Y_prob, Y_hat, CancerPosSource, _ = model(data)
                Y_hat = Y_hat.item()
                # CancerPosSource为小图注意力分数
                # 获取所有的坐标、以及癌小图的坐标及对应的注意力分数
                if isinstance(model, (CLAM_MB,)):
                    CancerPosSource = CancerPosSource[Y_hat]
                if isinstance(model, (MIL_fc,)):
                    CancerPosSource = CancerPosSource[:,1]
                CancerPosSource = CancerPosSource.view(-1, 1).cpu().numpy()
                # print('CancerPosSource:',CancerPosSource.shape,'modelType:',type(model))
                result_label = 'C1' if int(Y_hat) == 0 else 'C2'

                NonCancerPos = siteInfo[siteInfo['predictions'] == 0][['pos_x', 'pos_y']].to_numpy()
                CancerPos = siteInfo[siteInfo['predictions'] == 1][['pos_x', 'pos_y']].to_numpy()

                # 还需要获取原始切片level2 的缩略图    
                SlidePath = os.path.join(EXTERNAL_WSI_PATH,svsName.replace('.pt','.svs'))
                # (注意力热图,概略图,包含注意力分数最高/低的N张小图及位置的热图，包含注意力分数最高/低的N张小图及位置的概略图，N张注意分分数最高的小图（高低排序，N张注意力分数最低的小图（高低排序））
                heatmapSavePath = os.path.join(os.path.join(HeatmapSavePath,multi_instance,project),featurn,svsName.replace('.pt',f'_{result_label}'))

                attention_heatmap,summary_img,annotated_heatmap_img,annotated_summary_img,high_patch_list,\
                      low_patch_list,high_heatmap_patch_list, low_heatmap_patch_list = colorize_slide(SlidePath, NonCancerPos, CancerPos, CancerPosSource,heatmapSavePath)

                os.makedirs(heatmapSavePath,exist_ok=True)

                

                attention_heatmap.save(f"{heatmapSavePath}/attention_heatmap_{result_label}.png")
                summary_img.save(f"{heatmapSavePath}/summary_img_{result_label}.png")
                annotated_heatmap_img.save(f"{heatmapSavePath}/annotated_heatmap__{result_label}.png")
                annotated_summary_img.save(f"{heatmapSavePath}/annotated_summary_{result_label}.png")

                highPath = os.path.join(heatmapSavePath,'hign_wsi')
                lowPath = os.path.join(heatmapSavePath,'low_wsi')
                high_heatmapPath = os.path.join(heatmapSavePath,'hign_heatmap')
                low_heatmapPath = os.path.join(heatmapSavePath,'low_heatmap')
                os.makedirs(highPath,exist_ok=True)
                os.makedirs(lowPath,exist_ok=True)
                os.makedirs(high_heatmapPath,exist_ok=True)
                os.makedirs(low_heatmapPath,exist_ok=True)
                [v.save(f"{highPath}/high_{idx}.png") for idx,v in enumerate(high_patch_list)]
                [v.save(f"{lowPath}/low_{idx}.png") for idx,v in enumerate(low_patch_list)]

                [v.save(f"{high_heatmapPath}/high_{idx}.png") for idx,v in enumerate(high_heatmap_patch_list)]
                [v.save(f"{low_heatmapPath}/low_{idx}.png") for idx,v in enumerate(low_heatmap_patch_list)]
                # camImg.save("output_image.png")
                # break
