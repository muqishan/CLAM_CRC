import os
import pandas as pd
import ast
from utils.eval_utils import initiate_model
from utils.utils import get_simple_loader
from dataset_modules.dataset_generic import Generic_MIL_Dataset
from models.model_clam import CLAM_SB, CLAM_MB
from models.model_mil import MIL_fc, MIL_fc_mc
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from colorama import Fore, Style
import shutil

DEVICE = 'cuda:0'

def divideInstance(paths):
    # 根据路径list，将路径list划分为两个部分
    mil = []
    clam = []
    # txtPath = '/task_1_tumor_vs_normal_s22/experiment_task_1_tumor_vs_normal.txt'
    for path in paths:
        txtPath = os.path.join(path,f'{os.listdir(path)[0]}/experiment_task_1_tumor_vs_normal.txt')
        with open(txtPath, 'r', encoding='utf-8') as file:
            file_content = file.read()
            data_dict = ast.literal_eval(file_content)
            if data_dict['model_type'] == 'mil':
                 mil.append(path)
            else:
                clam.append(path)
    return  clam, mil

def summary_binary_confusion_matrix(model, loader, device=DEVICE):
    """
    只适用于二分类模型的评估，返回混淆矩阵。
    混淆矩阵中：
      - 行表示真实类别（C1、C2）
      - 列表示预测类别（C1、C2）
    """
    model.eval()

    all_labels = []
    all_preds = []

    for batch_idx, (data, label) in enumerate(loader):
        # 将数据和标签移动到指定 device
        data, label = data.to(device), label.to(device)
        
        # 不需要计算梯度
        with torch.no_grad():
            # 前向传播，获取模型预测
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        # label 和 Y_hat 转移到 CPU，并转换为 numpy
        all_labels.extend(label.cpu().numpy())
        all_preds.extend(Y_hat.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    # 将行、列分别命名为 [C1, C2]
    cm_df = pd.DataFrame(cm, index=['C1(真)', 'C2(真)'], columns=['C1(预测)', 'C2(预测)'])

    return cm_df

def loadModel(model_args,modelPath):
    # print('Init Model')    
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

def compute_youden_index(cm_df):
    """
    接受混淆矩阵(2x2 的 DataFrame)，计算并返回约登指数(Youden's Index)。
    """
    # 从 DataFrame 中取出 numpy array
    cm = cm_df.values  # shape (2, 2)
    
    # 取出 TP, TN, FP, FN
    TN, FP = cm[0,0], cm[0,1]
    FN, TP = cm[1,0], cm[1,1]
    
    # 计算 Sensitivity, Specificity
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan

    # 约登指数
    youden_index = sensitivity + specificity - 1
    if 'MSI' in PROJECT_NAMES:
        print('find best model use sens')
        return sensitivity
    return youden_index

def initModelArgs(path,featureModel):
    args_path = path.replace(os.path.basename(path),'experiment_task_1_tumor_vs_normal.txt')
    # print('args_path',args_path)
    with open(args_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
        data_dict = ast.literal_eval(file_content)
    data_dict['embed_dim'] = EMBED_DIM[featureModel]
    return data_dict

def evalClamModel(clamPath,EvalDataset,splitsLabelPath,featureModel):
    youden_indexs = []
    # acc_indexs = []
    for i in range(0,5): # 五折为5，其他折酌情替换
        if not  INDEPENDENT_TEST:
            split_dir = os.path.join('splits',splitsLabelPath)
            csv_path = '{}/splits_{}.csv'.format(split_dir, i)
            datasets = EvalDataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[DATASETS_ID[EVAL_Mode]]
            dataLoader = get_simple_loader(split_dataset)
        else:
            dataLoader = get_simple_loader(EvalDataset)
            print('启动独立测试集进行评估')
        modelPath = os.path.join(clamPath, os.listdir(clamPath)[0], 's_{}_checkpoint.pt'.format(i))
        # modelType = 'CLAM'
        print(modelPath)
        model_args = initModelArgs(modelPath,featureModel)
        model = loadModel(model_args,modelPath)     

        confusion_matrix = summary_binary_confusion_matrix(model,dataLoader)
        # 计算约登指数，
        youden_index = compute_youden_index(confusion_matrix)
        youden_indexs.append(youden_index)
        # 修改为使用ACC寻找最佳model 
        # acc_index = compute_accuracy(confusion_matrix)
        # acc_indexs.append(acc_index)
        # return youden_index
    youdenStd = np.std(youden_indexs) #约登指数-标准差
    youdenMean = np.mean(youden_indexs)   #约登指数-平均值
    # accMean = np.mean(acc_indexs) 
    return youdenMean

def compute_accuracy(cm_df):
    """
    接受混淆矩阵(2x2 的 DataFrame)，计算并返回 accuracy。
    """
    # 从 DataFrame 中取出 numpy array
    cm = cm_df.values  # shape (2, 2)

    # 取出 TN, FP, FN, TP
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    # 计算 accuracy
    total = TN + FP + FN + TP
    accuracy = (TP + TN) / total if total > 0 else np.nan

    return accuracy

def copy_all_contents(source_dir, target_dir):
    """
    将 source_dir 中的所有文件及子文件夹复制到 target_dir 中。
    如果 target_dir 不存在，则自动创建。
    """
    # 如果目标目录不存在，则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录下的所有文件和文件夹
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        target_path = os.path.join(target_dir, item)

        # 如果是文件，则使用 copy2 复制
        if os.path.isfile(item_path):
            shutil.copy2(item_path, target_path)
        # 如果是文件夹，则使用 copytree 复制整个目录
        elif os.path.isdir(item_path):
            # 注意：如果 target_path 已经存在，shutil.copytree() 默认会报错
            # 若 Python 版本 >= 3.8，可添加参数 dirs_exist_ok=True
            # shutil.copytree(item_path, target_path, dirs_exist_ok=True)
            shutil.copytree(item_path, target_path)


originLabelDict = {
                    'Angiogenesis':'label/Angiogenesis/Angiogenesis_test.csv',
                    'Autophagy':'label/Autophagy/Autophagy_test.csv',
                    'Stemness':'label/Stemness/Stemness_test.csv',
                    'MSI':'label/MSI/MSI_test_1.csv',
                    'TMB':'label/TMB/TMB_test_1.csv',
                    # 'Acetylation':'label/Acetylation/Acetylation_test.csv'
                }
splitsLabelDict = {
                    'Angiogenesis':'Angiogenesis',
                    'Autophagy':'Autophagy',
                    'Stemness':'Stemness',           
                    'MSI':'MSI',   
                    'TMB':'TMB',   
                    # 'Acetylation':'Acetylation'
                }


if __name__ == "__main__":
    BASE_PATH = 'result'
    # independent
    INDEPENDENT_TEST = True # 若启用独立测试集，则模型只会在指定的同一批数据集中进行评估
    # PROJECT_NAMES = ['Stemness_1'] #'Angiogenesis',
    # PROJECT_NAMES = ['Angiogenesis', 'Autophagy','Stemness'] #'Angiogenesis',
    PROJECT_NAMES = ['MSI', 'TMB'] #'Angiogenesis',
    FEATURE_MODELS = ['CTrans','uni','resnet']
    MULTI_INSTANCE_MODELS = ['clam_sb','clam_mb','mil']
    DATASETS_ID = {'train': 0, 'val': 1, 'test': 2, 'all': -1}
    os.makedirs('best_result',exist_ok=True)

    EMBED_DIM = {
                    'uni': 1024,
                    'CTrans':768,
                    'resnet':1024    
                }

    EVAL_Mode = 'test' # 默认只评估测试集
    # 目的，分别寻找mil和clam的最佳模型
    # 最佳模型的定义： 约登指数

    retult_bestmodel_mil = []
    retult_bestmodel_clam = []
    for PROJECT in PROJECT_NAMES:
        projectPath = os.path.join(BASE_PATH,PROJECT)
        # 获取待检测的数据集标签信息   (原始的标签信息->./label;五折后的标签信息->./splits)
        originLabelPath = originLabelDict[PROJECT]
        splitsLabelPath = splitsLabelDict[PROJECT]
        for featureModel in FEATURE_MODELS:
            featureModelPath = os.path.join(projectPath,featureModel)
            parameterModelspath = [os.path.join(featureModelPath,i) for i in os.listdir(featureModelPath)]
            clamPaths,milPaths = divideInstance(parameterModelspath)
            assert len(milPaths) > 0 ,' No { mil } training records found'
            assert len(clamPaths) > 0 ,' No { clam-mil } training records found'
            # 获取待检测的数据集标签信息、特征提取的数据集信息，不同模型特征提取后的数据信息->./RESULTS_DIRECTORY/uni)
            dataPath = os.path.join('RESULTS_DIRECTORY',featureModel)
            EvalDataset = Generic_MIL_Dataset(csv_path = originLabelPath,
                            data_dir= dataPath,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'C1':0, 'C2':1},
                            patient_strat=False,
                            ignore=[])  
            yuedenMeanMax = -1
            bestModelPath = ''
            for clamPath in clamPaths:
                youdenMean = evalClamModel(clamPath,EvalDataset,splitsLabelPath,featureModel)
                if youdenMean > yuedenMeanMax:
                    bestModelPath = clamPath
                    yuedenMeanMax = youdenMean
            # print(Fore.GREEN + "bestModelPath: " + bestModelPath + Style.RESET_ALL)
            retult_bestmodel_clam.append(bestModelPath)
            milYuedenMeanMax = -1
            milBestModelPath = ''
            for milPath in milPaths:
                youdenMean = evalClamModel(milPath,EvalDataset,splitsLabelPath,featureModel)
                if youdenMean > milYuedenMeanMax:
                    milBestModelPath = milPath
                    milYuedenMeanMax = youdenMean
            retult_bestmodel_mil.append(milBestModelPath)
            # print(featureModelPath,originLabelPath,splitsLabelPath,dataPath)
            # 加载待评估数据集，-> 
            # 分为两类，mil和clam，分别计算其约登指数，求平均约登指数，且标准差小于0.1
    # 将bestmodel移动到指定文件夹
    print(retult_bestmodel_clam)
    print(retult_bestmodel_mil)

    for bestmodel_clam in retult_bestmodel_clam:
        copy_all_contents(bestmodel_clam, os.path.join('best_result/clam',bestmodel_clam.split('/')[-3],bestmodel_clam.split('/')[-2]))

    for bestmodel_mil in retult_bestmodel_mil:
        copy_all_contents(bestmodel_mil, os.path.join('best_result/mil',bestmodel_mil.split('/')[-3],bestmodel_mil.split('/')[-2]))