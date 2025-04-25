'''
结肠癌自噬
1.uni
    uni_fold1
        混淆矩阵 
        AUC\ROC曲线                        融合
        注意力热图(待测样本的N张注意力热图)  只取bestMOdel
        待测样本的标签                      只取bestMOdel
        准确率（ACC, Accuracy）、灵敏度（SEN, Sensitivity）、特异度（SPE, Specificity）、F1值（F1, F1-score）、约登指数（YI, Youden's Index）
        NPV（NPV, Negative Predictive Value）、PPV（PPV, Positive Predictive Value）
    uni_fold2
    uni_fold3
    uni_fold4
    uni_fold5
2.ctrans
3.resnet
'''
from utils.utils import get_simple_loader,get_simple_loader_slide
import math
import os
import pandas as pd
import ast
from utils.eval_utils import initiate_model
from utils.utils import get_simple_loader
from dataset_modules.dataset_generic import Generic_MIL_Dataset,Generic_MIL_Dataset_PP
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
from sklearn.metrics import confusion_matrix, roc_curve, auc
plt.rcParams['font.family'] = 'Times New Roman'
DEVICE = 'cuda:0'


def initModelArgs(path,featureModel):
    args_path = path.replace(os.path.basename(path),'experiment_task_1_tumor_vs_normal.txt')
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

def ensure_roc_start_end(fpr, tpr):
    """
    若开头不是 (0,0) 就插入，
    若结尾不是 (1,1) 也插入。
    """
    # 开头
    if fpr[0] != 0.0 or tpr[0] != 0.0:
        fpr = np.insert(fpr, 0, 0.0)
        tpr = np.insert(tpr, 0, 0.0)
    # 结尾
    if fpr[-1] != 1.0 or tpr[-1] != 1.0:
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)
    return fpr, tpr

def remove_consecutive_duplicates(x, y):
    """
    移除 (x[i], y[i]) == (x[i-1], y[i-1]) 这种相邻重复点，
    以防止ROC序列里出现多次 (0,0) 或其他重复坐标。
    """
    new_x = [x[0]]
    new_y = [y[0]]
    for i in range(1, len(x)):
        if x[i] == new_x[-1] and y[i] == new_y[-1]:
            # 如果和上一个点一模一样，就跳过
            continue
        new_x.append(x[i])
        new_y.append(y[i])
    return np.array(new_x), np.array(new_y)

def format_auc_as_pct(auc_value):
    """把 0.xx 形式的 auc 转成 xx.xx% 字符串"""
    return f"{auc_value * 100:.2f}%"

def pct2(value):
    """把 0.xx 转成 xx.xx% 的字符串。"""
    return f"{value*100:.2f}%"



# plt.rcParams['axes.edgecolor'] = 'lightgray'   # 坐标轴线颜色
# plt.rcParams['axes.labelcolor'] = 'lightgray'  # 坐标轴标签文字颜色
# plt.rcParams['xtick.color'] = 'lightgray'      # x轴刻度文字颜色
# plt.rcParams['ytick.color'] = 'lightgray'      # y轴刻度文字颜色

def proportion_confidence_interval(p, n, z=1.96):
    """
    给定比例 p（0~1），样本量 n，返回 (lower, upper).
    使用 z * sqrt( p(1-p)/n ) 的正态近似
    """
    if n == 0:
        return (0, 0)
    se = math.sqrt(p * (1 - p) / n)
    lower = p - z * se
    upper = p + z * se
    # 修正下界上界至 [0,1]
    lower = max(0, lower)
    upper = min(1, upper)
    return (lower, upper)

def best_fold_ci_str(p, n):
    """
    p: (0,1) 之间的小数，表示指标值
    n: 该指标的分母样本量
    return:  "(xx.xx%, n/N, 95%CI[xx.xx% - xx.xx%])"
    """
    import math
    if n == 0:
        return "N=0"  # 避免分母=0时报错

    # 计算 95%CI (Wald)
    z = 1.96
    se = math.sqrt(p * (1 - p) / n)
    lower = max(0, p - z * se)
    upper = min(1, p + z * se)

    p_100 = p * 100
    l_100 = lower * 100
    u_100 = upper * 100
    return f"{p_100:.2f}%, {n}, 95%CI[{l_100:.2f}% - {u_100:.2f}%]"

def evalClamModel(clamPath, EvalDataset, splitsLabelPath, featureModel, 
                  process_name, method_name,metrics_store):

    for i in range(5):
        # === 1) 数据准备 & 模型加载 ===
        if not  INDEPENDENT_TEST:
            split_dir = os.path.join('splits',splitsLabelPath)
            csv_path = '{}/splits_{}.csv'.format(split_dir, i)
            datasets = EvalDataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[DATASETS_ID[EVAL_Mode]]
            dataLoader = get_simple_loader_slide(split_dataset)
        else:
            dataLoader = get_simple_loader_slide(EvalDataset)
            print('启动独立测试集进行评估')

        modelPath = os.path.join(clamPath, 's_{}_checkpoint.pt'.format(i))
        print(f"Eval fold {i} => model path: {modelPath}")
        model_args = initModelArgs(modelPath, featureModel)
        model = loadModel(model_args, modelPath)
        
        # === 2) 推理收集所有样本的标签/预测/概率 ===
        all_labels_list = []
        all_preds_list  = []
        all_probs_list  = []

        for batch_idx, (data, label,slide_id) in enumerate(dataLoader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            with torch.no_grad():
                logits, Y_prob, Y_hat, _, results_dict = model(data)
            all_labels_list.append(label.cpu().numpy())
            all_preds_list.append(Y_hat.cpu().numpy())
            all_probs_list.append(Y_prob.cpu().numpy())

        all_labels = np.concatenate(all_labels_list, axis=0)   # shape=(N,)
        all_preds  = np.concatenate(all_preds_list, axis=0)    # shape=(N,)
        all_probs  = np.concatenate(all_probs_list, axis=0)    # shape=(N,2)

        # === 3) 混淆矩阵(默认"正类"=C2=1) ===
        cm = confusion_matrix(all_labels, all_preds)
        #     Pred=C1  Pred=C2
        # C1   cm[0,0] cm[0,1]
        # C2   cm[1,0] cm[1,1]
        cm = cm.astype(int)

        TN, FP = cm[0,0], cm[0,1]
        FN, TP = cm[1,0], cm[1,1]

        # (a) 不分 C1/C2 => Accuracy, Youden
        total_samples = cm.sum()  # (TP+TN+FP+FN)
        
       # (a) Accuracy
        acc = (TP + TN) / total_samples
        acc_n = total_samples
        acc_x = TP + TN
        # x= (TP+TN)

        # (b) Sensitivity(C2) = TP / (TP+FN)
        sens_c2 = TP/(TP+FN) if (TP+FN)!=0 else 0
        sens_c2_n = (TP+FN)
        sens_c2_x = TP  

        # (c) Specificity(C2) = TN / (TN+FP)
        spec_c2 = TN/(TN+FP) if (TN+FP)!=0 else 0
        spec_c2_n = (TN+FP)
        spec_c2_x = TN

        # (d) PPV(C2) = TP / (TP+FP)
        ppv_c2 = TP/(TP+FP) if (TP+FP)!=0 else 0
        ppv_c2_n = (TP+FP)
        ppv_c2_x = TP
        # (e) NPV(C2) = TN / (TN+FN)
        npv_c2 = TN/(TN+FN) if (TN+FN)!=0 else 0
        npv_c2_n = (TN+FN)
        npv_c2_x = TN
        # (f) F1(C2) = 2 * (precision * recall)/(precision + recall)
        #   precision=ppv_c2, recall=sens_c2
        f1_c2 = 0
        if (ppv_c2 + sens_c2) > 0:
            f1_c2 = 2 * (ppv_c2 * sens_c2) / (ppv_c2 + sens_c2)

        # (g) 约登指数 (Youden's Index) = Sens(C2) + Spec(C2) - 1
        yi = sens_c2 + spec_c2 - 1
 
        # (h) 针对 C1(阳性)，类似地写
        #   TP_c1=cm[0,0], FN_c1=cm[0,1], FP_c1=cm[1,0], TN_c1=cm[1,1]
        TP_c1 = cm[0,0]
        FN_c1 = cm[0,1]
        FP_c1 = cm[1,0]
        TN_c1 = cm[1,1]

        #   Sens(C1) = TP_c1/(TP_c1+FN_c1)
        sens_c1 = TP_c1/(TP_c1+FN_c1) if (TP_c1+FN_c1)!=0 else 0
        sens_c1_n = TP_c1+FN_c1
        sens_c1_x = TP_c1
        #   Spec(C1) = TN_c1/(TN_c1+FP_c1)
        spec_c1 = TN_c1/(TN_c1+FP_c1) if (TN_c1+FP_c1)!=0 else 0
        spec_c1_n = TN_c1+FP_c1
        spec_c1_x = TN_c1

        #   PPV(C1) = TP_c1/(TP_c1+FP_c1)
        ppv_c1 = TP_c1/(TP_c1+FP_c1) if (TP_c1+FP_c1)!=0 else 0
        ppv_c1_n = TP_c1+FP_c1
        ppv_c1_x = TP_c1

        #   NPV(C1) = TN_c1/(TN_c1+FN_c1)
        npv_c1 = TN_c1/(TN_c1+FN_c1) if (TN_c1+FN_c1)!=0 else 0
        npv_c1_n = TN_c1+FN_c1
        npv_c1_x = TN_c1

        #   F1(C1)
        f1_c1 = 0
        if (ppv_c1 + sens_c1) > 0:
            f1_c1 = 2 * (ppv_c1 * sens_c1)/(ppv_c1 + sens_c1)
        # === 4) 分别计算 ROC/AUC ===

        fpr_c2, tpr_c2, _ = roc_curve(all_labels, all_probs[:,1], pos_label=1)
        # auc_c2 = auc(fpr_c2, tpr_c2)
        fpr_c1, tpr_c1, _ = roc_curve(all_labels, all_probs[:,0], pos_label=0)
        fpr_c2, tpr_c2 = ensure_roc_start_end(fpr_c2, tpr_c2)
        # 移除相邻重复点
        fpr_c2, tpr_c2 = remove_consecutive_duplicates(fpr_c2, tpr_c2)
        # AUC
        auc_c2 = auc(fpr_c2, tpr_c2)
        n_plus_C2 = sum(all_labels == 1)
        n_minus_C2 = sum(all_labels != 1)
        auc_c2_n = n_plus_C2 * n_minus_C2

        fpr_c1, tpr_c1 = ensure_roc_start_end(fpr_c1, tpr_c1)
        # 移除相邻重复点
        fpr_c1, tpr_c1 = remove_consecutive_duplicates(fpr_c1, tpr_c1)
        # AUC
        auc_c1 = auc(fpr_c1, tpr_c1)
        n_plus_C1 = sum(all_labels == 0)
        n_minus_C1 = sum(all_labels != 0)
        auc_c1_n = n_plus_C1 * n_minus_C1

        if process_name not in metrics_store:
            metrics_store[process_name] = {}

        if featureModel not in metrics_store[process_name]:
            metrics_store[process_name][featureModel] = {}

        if method_name not in metrics_store[process_name][featureModel]:
            metrics_store[process_name][featureModel][method_name] = {}

        fold_key = f"fold_{i}"
        metrics_store[process_name][featureModel][method_name][fold_key] = {
            # "ACC": acc,
            # 'ACC_N': acc_n,
            # 'ACC_X': acc_x,
            # "ACC_95%CI": proportion_confidence_interval(acc,acc_n),
            f'ROC({label_name_C1})':(fpr_c1, tpr_c1, auc_c1),
            f'ROC({label_name_C2})':(fpr_c2, tpr_c2, auc_c2),
            "ACC":f"{acc*100:.2f}% ({acc_x}/{acc_n}, ({proportion_confidence_interval(acc,acc_n)[0]*100:.2f}-{proportion_confidence_interval(acc,acc_n)[1]*100:.2f}%)",

            "Youden": f"{yi*100:.2f}%",
            

            # "AUC(C2)": auc_c2,
            # 'AUC(C2)_N': auc_c2_n,
            # "AUC(C2)_95%CI":proportion_confidence_interval(auc_c2,auc_c2_n),
            f"AUC({label_name_C2})":f"{auc_c2*100:.2f}% ({proportion_confidence_interval(auc_c2,auc_c2_n)[0]*100:.2f}-{proportion_confidence_interval(auc_c2,auc_c2_n)[1]*100:.2f}%)",

            # "Sensitivity(C2)": sens_c2,
            # "Sens_C2_N" : sens_c2_n,
            # "Sens_C2_X" : sens_c2_x,
            # "Sensitivity(C2)_95%CI": proportion_confidence_interval(sens_c2,sens_c2_n),
            f"Sensitivity({label_name_C2})": f"{sens_c2*100:.2f}% ({sens_c2_x}/{sens_c2_n}, {proportion_confidence_interval(sens_c2,sens_c2_n)[0]*100:.2f}-{proportion_confidence_interval(sens_c2,sens_c2_n)[1]*100:.2f}%)",

            # "Specificity(C2)": spec_c2,
            # "Spec_C2_N": spec_c2_n,
            # "Spec_C2_X": spec_c2_x,
            # "Specificity(C2)_95%CI": proportion_confidence_interval(spec_c2,spec_c2_n),
            f"Specificity({label_name_C2})": f"{spec_c2*100:.2f}% ({spec_c2_x}/{spec_c2_n}, {proportion_confidence_interval(spec_c2,spec_c2_n)[0]*100:.2f}-{proportion_confidence_interval(spec_c2,spec_c2_n)[1]*100:.2f}%)",

            # "PPV(C2)": ppv_c2,
            # "PPV(C2)_N": ppv_c2_n,
            # "PPV(C2)_X": ppv_c2_x,
            # "PPV(C2)_95%CI": proportion_confidence_interval(ppv_c2,ppv_c2_n),
            f"PPV({label_name_C2})": f"{ppv_c2*100:.2f}% ({ppv_c2_x}/{ppv_c2_n}, {proportion_confidence_interval(ppv_c2,ppv_c2_n)[0]*100:.2f}-{proportion_confidence_interval(ppv_c2,ppv_c2_n)[1]*100:.2f}%)",

            # "NPV(C2)": npv_c2,
            # "NPV(C2)_N": npv_c2_n,
            # "NPV(C2)_X": npv_c2_x,
            # "NPV(C2)_95%CI": proportion_confidence_interval(npv_c2,npv_c2_n),
            f"NPV({label_name_C2})": f"{npv_c2*100:.2f}% ({npv_c2_x}/{npv_c2_n}, {proportion_confidence_interval(npv_c2,npv_c2_n)[0]*100:.2f}-{proportion_confidence_interval(npv_c2,npv_c2_n)[1]*100:.2f}%)",

            f"F1({label_name_C2})": f"{f1_c2*100:.2f}%",

            # "AUC(C1)": auc_c1,
            # "AUC(C1)_N":auc_c1_n,
            # "AUC(C1)_95%CI":proportion_confidence_interval(auc_c1,auc_c1_n),
           f"AUC({label_name_C1})":f"{auc_c1*100:.2f}% ({proportion_confidence_interval(auc_c1,auc_c1_n)[0]*100:.2f}-{proportion_confidence_interval(auc_c1,auc_c1_n)[1]*100:.2f}%)",

            # "Sensitivity(C1)": sens_c1,
            # "Sens_C1_N" : sens_c1_n,
            # "Sens_C1_X" : sens_c1_x,
            # "Sensitivity(C1)_95%CI": proportion_confidence_interval(sens_c1,sens_c1_n),
            f"Sensitivity(C{label_name_C1}1)": f"{sens_c1*100:.2f}% ({sens_c1_x}/{sens_c1_n}, {proportion_confidence_interval(sens_c1,sens_c1_n)[0]*100:.2f}-{proportion_confidence_interval(sens_c1,sens_c1_n)[1]*100:.2f}%)",

            # "Specificity(C1)": spec_c1,
            # "Spec_C1_N": spec_c1_n,
            # "Spec_C1_X": spec_c1_x,
            # "Specificity(C1)_95%CI": proportion_confidence_interval(spec_c1,spec_c1_n),
            f"Specificity({label_name_C1})": f"{spec_c1*100:.2f}% ({spec_c1_x}/{spec_c1_n}, {proportion_confidence_interval(spec_c1,spec_c1_n)[0]*100:.2f}-{proportion_confidence_interval(spec_c1,spec_c1_n)[1]*100:.2f}%)",

            # "PPV(C1)": ppv_c1,
            # "PPV(C1)_N": ppv_c1_n,
            # "PPV(C1)_X": ppv_c1_x,
            # "PPV(C1)_95%CI": proportion_confidence_interval(ppv_c1,ppv_c1_n),
            f"PPV({label_name_C1})": f"{ppv_c1*100:.2f}% ({ppv_c1_x}/{ppv_c1_n}, {proportion_confidence_interval(ppv_c1,ppv_c1_n)[0]*100:.2f}-{proportion_confidence_interval(ppv_c1,ppv_c1_n)[1]*100:.2f}%)",

            # "NPV(C1)": npv_c1,
            # "NPV(C1)_N": npv_c1_n,
            # "NPV(C1)_X": npv_c1_x,
            # "NPV(C1)_95%CI": proportion_confidence_interval(npv_c1,npv_c1_n),
            f"NPV({label_name_C1})": f"{npv_c1*100:.2f}% ({npv_c1_x}/{npv_c1_n}, {proportion_confidence_interval(npv_c1,npv_c1_n)[0]*100:.2f}-{proportion_confidence_interval(npv_c1,npv_c1_n)[1]*100:.2f}%)",

            f"F1({label_name_C1})": f"{f1_c1*100:.2f}%",
        }
        
label_name_C1 = 'Non-MSI-H'
label_name_C2 = 'MSI-H'
save_dir = 'eval_result_MSI'
os.makedirs(save_dir, exist_ok=True)
BASE_PATH = 'best_result'
MULTI_INSTANCE_MODELS = ['clam','mil']
# PROJECT_NAMES = ['Angiogenesis', 'Autophagy','Stemness']
# PROJECT_NAMES = ['MSI']
PROJECT_NAMES = ['TMB']
# PROJECT_NAMES = ['Acetylation'] 
FEATURE_MODELS = ['CTrans','uni','resnet']
DATASETS_ID = {'train': 0, 'val': 1, 'test': 2, 'all': -1}
INDEPENDENT_TEST = True
EMBED_DIM = {
                'uni': 1024,
                'CTrans':768,
                'resnet':1024    
            }
EVAL_Mode = 'test' # 默认只需要对test进行评估即可
originLabelDict = {
                    'Angiogenesis':'label/Angiogenesis/Angiogenesis_test.csv',
                    'Autophagy':'label/Autophagy/Autophagy_test.csv',
                    'Stemness':'label/Stemness/Stemness_test.csv',
                    'MSI':'label/MSI/MSI_test_1.csv',
                    'TMB':'label/TMB/TMB_test_1.csv',
                    # 'Acetylation':'label/Acetylation/Acetylation_test.csv'
                }
metrics_dict = {}

for multi_instance in MULTI_INSTANCE_MODELS:  # method: mil/clam
    base_multi_instance_path = os.path.join(BASE_PATH, multi_instance)
    for project in PROJECT_NAMES:  # process: Angiogenesis/Autophagy/Stemness
        project_path = os.path.join(base_multi_instance_path, project)
        for featurn in FEATURE_MODELS:  # model: uni/resnet/CTrans
            featurn_models = os.path.join(os.path.join(project_path, featurn),os.listdir(os.path.join(project_path, featurn))[0])

            originLabelPath = originLabelDict[project]
            dataPath =  os.path.join('RESULTS_DIRECTORY', featurn)

            EvalDataset = Generic_MIL_Dataset_PP(
                csv_path=originLabelPath,
                data_dir=dataPath,
                shuffle=False, 
                print_info=True,
                label_dict={'C1':0, 'C2':1},
                patient_strat=False,
                ignore=[]
            )
            

            evalClamModel(
                clamPath=featurn_models,
                EvalDataset=EvalDataset,
                splitsLabelPath=project,
                featureModel=featurn,
                process_name=project,
                method_name=multi_instance,
                metrics_store=metrics_dict
            )
color_map = {
    "CTrans_clam": "C0",
    "CTrans_mil":  "C1",
    "uni_clam":    "C2",
    "uni_mil":     "C3",
    "resnet_clam": "C4",
    "resnet_mil":  "C5"
}
StandardName = {
    'uni':'UNI',
    'clam':'CLAM',
    'CTrans':'CTrans',
    'resnet':'ResNet50',
    'mil':'Classic MIL'
}
# 针对每个任务（process_name），遍历每个折（fold_0 到 fold_4）生成合成 ROC 图
for process_name in PROJECT_NAMES:
    for i in range(5):  # i 表示折号 0~4
        plt.figure(figsize=(8,6), dpi=400)
        # 遍历所有特征提取方法和多实例方法组合
        for feat in FEATURE_MODELS:
            for method in MULTI_INSTANCE_MODELS:
                # 构造模型组合键，如 "CTrans_clam"
                model_key = f"{StandardName[feat]}+{StandardName[method]}"
                # 检查当前任务下是否有该组合以及该折的数据                   
                fprC1, tprC1, auc_valC1 = metrics_dict[process_name][feat][method][f"fold_{i}"][f"ROC({label_name_C1})"]
                del metrics_dict[process_name][feat][method][f"fold_{i}"][f"ROC({label_name_C1})"]
                plt.plot(fprC1, tprC1, color=color_map.get(model_key, None), lw=2,
                            label=f"{model_key} (AUC={auc_valC1*100:.2f}%)")
        # 绘制随机猜测的对角线
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Chance")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title(f"{process_name} - Fold {i} Combined ROC", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        # 保存图像，文件名格式：{process_name}_fold_{i+1}_combined_ROC.png
        plt.savefig(f"{save_dir}/{process_name}_fold_{i}_combined_ROC_{label_name_C1}.png", bbox_inches='tight')
        plt.close()

for process_name in PROJECT_NAMES:
    for i in range(5):  # i 表示折号 0~4
        plt.figure(figsize=(8,6), dpi=400)
        # 遍历所有特征提取方法和多实例方法组合
        for feat in FEATURE_MODELS:
            for method in MULTI_INSTANCE_MODELS:
                # 构造模型组合键，如 "CTrans_clam"
                model_key = f"{StandardName[feat]}+{StandardName[method]}"
                # 检查当前任务下是否有该组合以及该折的数据
                fprC2, tprC2, auc_valC2 = metrics_dict[process_name][feat][method][f"fold_{i}"][f"ROC({label_name_C2})"]
                del metrics_dict[process_name][feat][method][f"fold_{i}"][f"ROC({label_name_C2})"]
                plt.plot(fprC2, tprC2, color=color_map.get(model_key, None), lw=2,
                            label=f"{model_key} (AUC={auc_valC2*100:.2f}%)")
        # 绘制随机猜测的对角线
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Chance")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title(f"{process_name} - Fold {i} Combined ROC", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        # 保存图像，文件名格式：{process_name}_fold_{i+1}_combined_ROC.png
        plt.savefig(f"{save_dir}/{process_name}_fold_{i}_combined_ROC_{label_name_C2}.png", bbox_inches='tight')
        plt.close()


rows = []  # 用于存放扁平化后的数据
for project_name, feat_dict in metrics_dict.items():
    for feat, method_dict in feat_dict.items():
        for method, fold_dict in method_dict.items():
            for fold_i, metric_vals in fold_dict.items():
                row = {
                    "Project": project_name,
                    "FeatureModel": feat,
                    "Method": method,
                    "Fold": fold_i
                }
                # 把 metric_vals dict 合并进来
                row.update(metric_vals)
                rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(f"{save_dir}/all_eval_results.csv", index=False, encoding="utf-8-sig")


