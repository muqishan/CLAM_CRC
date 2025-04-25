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
import seaborn as sns
from torch.utils.data import DataLoader, Dataset, SequentialSampler

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
        return data, os.path.basename(file_path)

def collate_fn_with_filename(batch):
    """
    自定义 collate_fn，处理特征矩阵和文件名。
    """
    features = torch.cat([item[0] for item in batch], dim = 0)
    filenames = [item[1] for item in batch][0]
    return features, filenames

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
MULTI_INSTANCE_MODELS = ['clam','mil']
PROJECT_NAMES = ['Angiogenesis', 'Autophagy','Stemness','MSI','TMB']
FEATURE_MODELS = ['CTrans','uni','resnet']
EXTERNAL_DATA_PATH = 'ExternalDataCrc/wsi/20250327/RESULTS_DIRECTORY/***/pt_files' #
EMBED_DIM = {
                'uni': 1024,
                'CTrans':768,
                'resnet':1024    
            }
resultSavePath = 'ExternalDataCrc/wsi/20250327/modelPredict'
os.makedirs(resultSavePath,exist_ok=True)
for multi_instance in MULTI_INSTANCE_MODELS:  # method: mil/clam
    base_multi_instance_path = os.path.join(BASE_PATH, multi_instance)
    for project in PROJECT_NAMES:  # process: Angiogenesis/Autophagy/Stemness
        project_path = os.path.join(base_multi_instance_path, project)
        for featurn in FEATURE_MODELS:  # model: uni/resnet/CTrans
            print(os.path.join(os.path.join(project_path, featurn),os.listdir(os.path.join(project_path, featurn))[0]))
            print( os.path.join(project_path, featurn, os.listdir(os.path.join(project_path, featurn))[0]))
            print('____________________________')
            modelRootPath = os.path.join(project_path, featurn, os.listdir(os.path.join(project_path, featurn))[0])
            modelConfig = initModelArgs(os.path.join(modelRootPath,'experiment_task_1_tumor_vs_normal.txt'),featurn)
            model = loadModel(modelConfig, os.path.join(modelRootPath,[i for i in os.listdir(modelRootPath) if i.endswith('.pt')][0]))

            ExternalDataPath = EXTERNAL_DATA_PATH.replace('***',featurn)
            ExternalDataPath = [os.path.join(ExternalDataPath,i) for i in os.listdir(ExternalDataPath)]
            dataset = PTFDataset(ExternalDataPath)

            # 获取数据加载器
            dataLoader = get_simple_loader(dataset, batch_size=1, num_workers=1)
            result = {
                'svsName':[],
                'Predict':[]
            }
            for batch_idx, (data, svsName) in enumerate(dataLoader):
                data = data.to(DEVICE)
                with torch.no_grad():
                    logits, Y_prob, Y_hat, _, results_dict = model(data)
                    result['svsName'].append(svsName)
                    result['Predict'].append('C1' if int(Y_hat) == 0 else 'C2')
            
                    # Y_hat就是类别名
            resultPath = os.path.join(resultSavePath,str(multi_instance)+'_'+str(project)+'_'+str(featurn)+'.csv')
            pd.DataFrame(result).to_csv(resultPath,index=False)
