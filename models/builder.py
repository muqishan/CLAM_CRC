import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


# device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# normalizer = torch.load('tools/macenko_normalizer_torch.pkl',map_location=device)
# normalizer = normalizer.to(device)


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
    
def load_Private_model(model_path):
    net = EfficientNetV2WithDropout() 
    checkpoint = torch.load(model_path)
    
    # 去掉 'module.' 前缀（如果存在）
    if 'module.' in list(checkpoint.keys())[0]:
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    # 加载模型权重
    net.load_state_dict(checkpoint)
    
    # 将分类层替换为恒等层
    net.efficientnet.classifier = nn.Identity()  # 这里移除分类器，替换为恒等层
    
    net.eval()  # 设置为评估模式
    return net


    return net  # Return the mode

def get_encoder(model_name, target_img_size=224):
    
    print('loading model checkpoint')
    if model_name == 'CTrans':
        print('modelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelname:','CTrans')
        from .model_CTransPath import ctranspath
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose(
            [
                # transforms.Resize(224),
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: normalizer(x)), 
                transforms.Normalize(mean = mean, std = std)
            ]
        )
        net = ctranspath()
        net.head = nn.Identity()
        td = torch.load('net/ctranspath.pth')
        net.load_state_dict(td['model'], strict=True)
        net.eval()
        return net, transform
    
    elif model_name == 'Private':
        print('modelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelname:','Private')
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        net = load_Private_model('net/Epoch_013_Acc:_97.3038.pth')  # 121 169 201 161
        # # net = ResNet34WithAttention()

        transform = transforms.Compose(
            [
                # transforms.Resize(224),
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: normalizer(x)), 
                transforms.Normalize(mean = mean, std = std)
            ]
        )
        return net, transform
    elif model_name == 'uni':
        print('modelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelname:','uni')
        # https://huggingface.co/MahmoodLab/UNI
        # uni的timm版本有问题，需要最新版本的timm
        net = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        # net.load_state_dict(torch.load('net/uni.bin', map_location="cpu"), strict=True)
        net.load_state_dict(torch.load('net/uni.bin'), strict=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose(
            [
                # transforms.Resize(224),
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: normalizer(x)), 
                transforms.Normalize(mean = mean, std = std)
            ]
        )
        return net,transform
    elif model_name == 'resnet':
        print('modelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelnamemodelname:','resnet')
        net  = TimmCNNEncoder()
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose(
            [
                # transforms.Resize(224),
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: normalizer(x)), 
                transforms.Normalize(mean = mean, std = std)
            ]
        )
        return net,transform


        # 通用非医学背景的特征提取

# def get_model(model_name,target_img_size=224):
#     if model_name == 'CTransPathModel':
#         from models.model_CTransPath import ctranspath
#         mean = (0.485, 0.456, 0.406)
#         std = (0.229, 0.224, 0.225)
#         transforms = transforms.Compose(
#             [
#                 transforms.Resize(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean = mean, std = std)
#             ]
#         )
#         net = ctranspath()
#         net.head = nn.Identity()
#         td = torch.load(model_path)
#         net.load_state_dict(td['model'], strict=True)
#         net.eval()
#         return net, img_transforms

#     def loadCTransPathModel(self,model_path):
#         from models.model_CTransPath import ctranspath
#         mean = (0.485, 0.456, 0.406)
#         std = (0.229, 0.224, 0.225)
#         self.transforms = transforms.Compose(
#             [
#                 transforms.Resize(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean = mean, std = std)
#             ]
#         )
#         net = ctranspath()
#         net.head = nn.Identity()
#         td = torch.load(model_path)
#         net.load_state_dict(td['model'], strict=True)
#         net.eval()
#         self.net = net.to(self.device)
