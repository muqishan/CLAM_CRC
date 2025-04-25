import h5py
import os
import torch
import torchvision.transforms as transforms
from openslide import OpenSlide
from PIL import Image
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


device = 'cuda:0'
class EfficientNetV2WithDropout(nn.Module):
    def __init__(self):
        super(EfficientNetV2WithDropout, self).__init__()
        # 使用 EfficientNetV2-S 作为基础模型
        self.efficientnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        
        # 获取特征提取部分输出的特征数
        num_ftrs = self.efficientnet.classifier[1].in_features

        # 在全连接层前添加一个额外的 Dropout 和 BatchNorm 层，防止过拟合
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.3),                 # 第一个 Dropout 层
            nn.BatchNorm1d(num_ftrs),        # 批归一化层
            nn.Dropout(0.5),                 # 第二个 Dropout 层
            nn.Linear(num_ftrs, 2)           # 最终二分类层
        )

    def forward(self, x):
        x = self.efficientnet.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # 自适应平均池化
        x = torch.flatten(x, 1)               # 展平
        x = self.efficientnet.classifier(x)   # 分类器
        return x

    
BATCH_SIZE = 512
patches_root_path = 'ExternalDataCrc/My_RESULTS_DIRECTORYdata2/patches'
PREDICT_PATH = patches_root_path.replace('patches','predicts')
wsi_root_paths = 'ExternalDataCrc/wsi/20mpp_2'
PATCHE_SIZE = 224
LEVEL = 0
# result_path = '/datasets/colon_cancer_datasets/ROIdatasets'
norm_mean = [0.485,0.456,0.406]
norm_std = [0.229,0.224,0.225]

pre_transform = transforms.Compose([
    # transforms.Resize((224, 224)),.
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean, std=norm_std),
])

model_path = 'net/Epoch_013_Acc:_97.3038.pth'

os.path.exists(PREDICT_PATH) or os.mkdir(PREDICT_PATH)

# post_transform = transforms.Normalize(norm_mean, norm_std)
# executor = ThreadPoolExecutor(max_workers=128)
def load_model(model_path):
    net = EfficientNetV2WithDropout() 
    checkpoint = torch.load(model_path, map_location=device)
    if 'module.' in list(checkpoint.keys())[0]:
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    # Load model weights
    net.load_state_dict(checkpoint)
    net = net.to(device)
    net.eval()

    return net  # Return the model


def batch_extract(data, batch_size=BATCH_SIZE):
    num_total = data.shape[0]  # 获取数组的总行数
    for start in range(0, num_total, batch_size):
        end = min(start + batch_size, num_total)  # 确保不会超出数组的边界
        yield data[start:end]  # 返回从start到end（不包括end）的数据片段


def process_svs_file(slide, batch_point, model, level,svs_name,tile_size=(PATCHE_SIZE, PATCHE_SIZE)):
    regions = []
    positions = []
    for x,y in batch_point:
        region = slide.read_region((x, y), level, tile_size).convert('RGB')
        regions.append(region)
        positions.append((x, y))
    # tiles = [Image.fromarray(np.array(r.convert("RGB"))) for r in regions]
    # tile_tensors = torch.stack([pre_transform(t) for t in tiles]).cuda()

    tile_tensors = torch.stack([pre_transform(r) for  r in regions]).cuda()
   
    # tile_tensors = pre_transform(tile_tensors)  # 归一化整个批量
    result = []
    with torch.no_grad():
        outputs = model(tile_tensors)
        probabilities = F.softmax(outputs, dim=1)
        prob_class1 = probabilities[:, 1]
        predictions = (prob_class1 > 0.5).long()
        # torch.cuda.empty_cache()
        for i, (pos_x, pos_y) in enumerate(positions):
            info = {
                'svs_name':svs_name,
                'pos_x':pos_x,
                'pos_y':pos_y,
                'predictions':int(predictions[i]),
                'prob':round(float(prob_class1[i]),4),
            }
            result.append(info)

    return result
       
model = load_model(model_path)

for patche in tqdm(os.listdir(patches_root_path)):
    wsi_path = os.path.join(wsi_root_paths,patche.replace('.h5','.svs'))
    patche_path = os.path.join(patches_root_path,patche)
    # 每次取batch_size个
    with h5py.File(patche_path, 'r') as file:
        # print(list(file.keys()))
        patche_list = file['coords'][:]
        slide = OpenSlide(wsi_path)
        svs_name = patche[:23]
        csv_name = svs_name+'.csv'
        results = []
        for batch_point in batch_extract(patche_list):
            r = process_svs_file(slide,batch_point,model,LEVEL,svs_name)
            results.extend(r)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(PREDICT_PATH,csv_name),index=False)

#nohup python checkClamRoi.py &> output.log &