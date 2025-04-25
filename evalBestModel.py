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
from scipy.stats import chi2
import shutil
import math
import os
import pandas as pd
import ast
from utils.eval_utils import initiate_model
from utils.utils import get_simple_loader,get_simple_loader_slide
from dataset_modules.dataset_generic import Generic_MIL_Dataset_PP
from models.model_clam import CLAM_SB, CLAM_MB
from models.model_mil import MIL_fc, MIL_fc_mc
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy.stats import chi2_contingency,fisher_exact
from math import sqrt
from torch.utils.data import ConcatDataset

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

def evalClamModel(clamPath, EvalDataset, splitsLabelPath, featureModel, process_name, method_name):
    folds_acc = []
    folds_yi  = []
    folds_sens = []
    folds_auc_c2= []
    folds_sens_c2= []
    folds_spec_c2= []
    folds_ppv_c2= []
    folds_npv_c2= []
    folds_f1_c2 = []
    folds_auc_c1= []
    folds_sens_c1= []
    folds_spec_c1= []
    folds_ppv_c1= []
    folds_npv_c1= []
    folds_f1_c1 = []

    folds_acc_n = []
    folds_acc_x = []
    folds_sens_c2_n = []
    folds_sens_c2_x = []
    folds_spec_c2_n = []
    folds_spec_c2_x = []
    folds_ppv_c2_n  = []
    folds_ppv_c2_x  = []
    folds_npv_c2_n  = []
    folds_npv_c2_x  = []
    folds_auc_c2_n  = []
    folds_f1_c2_n   = []

    folds_sens_c1_n = []
    folds_sens_c1_x = []
    folds_spec_c1_n = []
    folds_spec_c1_x = []
    folds_ppv_c1_n  = []
    folds_ppv_c1_x  = []
    folds_npv_c1_n  = []
    folds_npv_c1_x  = []
    folds_auc_c1_n  = []
    folds_f1_c1_n   = []

    # 额外存储：每折的 case-level 结果
    fold_case_results = {}

    # 新增：记录每折的ROC数据（以C2为例），格式：(fpr, tpr, auc)
    folds_roc_data_C1 = []
    folds_roc_data_C2 = []
    for i in range(5):
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

        all_labels_list = []
        all_preds_list  = []
        all_probs_list  = []
        all_ids_list    = []  # 存放 sample_id

        # ---------- 推理阶段，保存逐例 ----------
        for batch_idx, (data, label,slide_id) in enumerate(dataLoader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            with torch.no_grad():
                logits, Y_prob, Y_hat, _, results_dict = model(data)

            all_labels_list.append(label.cpu().numpy())
            all_preds_list.append(Y_hat.cpu().numpy())
            all_probs_list.append(Y_prob.cpu().numpy())
            all_ids_list.append(slide_id)
        
        all_labels = np.concatenate(all_labels_list, axis=0)
        all_preds  = np.concatenate(all_preds_list, axis=0)
        all_probs  = np.concatenate(all_probs_list, axis=0)

        # 保存 case-level 信息
        fold_case_results[i] = list(zip(all_ids_list, all_labels, [ar[0] for ar in all_preds.tolist()]))

        cm = confusion_matrix(all_labels, all_preds)
        cm = cm.astype(int)
        TN, FP = cm[0,0], cm[0,1]
        FN, TP = cm[1,0], cm[1,1]
        total_samples = cm.sum()

        # 绘制混淆矩阵
        plt.figure(figsize=(8,6), dpi=400)
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                    xticklabels=[f"{label_name_C1}",f"{label_name_C2}"], yticklabels=[f"{label_name_C1}",f"{label_name_C2}"],annot_kws={"size":20})
        
        plt.title(f"Confusion Matrix (Fold={i})",fontsize=20)
        plt.xlabel("Predicted",fontsize=20)
        plt.ylabel("True",fontsize=20)
        cm_save_path = f"{save_dir}/confusionMatrix_{splitsLabelPath}_{featureModel}_{method_name}_fold_{i}.png"
        plt.savefig(cm_save_path, bbox_inches='tight')
        plt.close()

        # Accuracy
        acc = (TP + TN) / total_samples
        folds_acc.append(acc)
        folds_acc_n.append(total_samples)
        folds_acc_x.append(TP + TN)

        # Sens(C2)
        sens_c2 = TP / (TP+FN) if (TP+FN)>0 else 0
        folds_sens_c2.append(sens_c2)
        folds_sens_c2_n.append(TP+FN)
        folds_sens_c2_x.append(TP)
        folds_sens
        # Spec(C2)
        spec_c2 = TN / (TN+FP) if (TN+FP)>0 else 0
        folds_spec_c2.append(spec_c2)
        folds_spec_c2_n.append(TN+FP)
        folds_spec_c2_x.append(TN)

        # PPV(C2)
        ppv_c2 = TP / (TP+FP) if (TP+FP)>0 else 0
        folds_ppv_c2.append(ppv_c2)
        folds_ppv_c2_n.append(TP+FP)
        folds_ppv_c2_x.append(TP)

        # NPV(C2)
        npv_c2 = TN / (TN+FN) if (TN+FN)>0 else 0
        folds_npv_c2.append(npv_c2)
        folds_npv_c2_n.append(TN+FN)
        folds_npv_c2_x.append(TN)

        # F1(C2)
        f1_c2 = 0
        if ppv_c2+sens_c2>0:
            f1_c2 = 2*ppv_c2*sens_c2/(ppv_c2+sens_c2)
        folds_f1_c2.append(f1_c2)

        # Youden
        yi = sens_c2 + spec_c2 - 1
        folds_yi.append(yi)

        # C1指标
        TP_c1 = cm[0,0]
        FN_c1 = cm[0,1]
        FP_c1 = cm[1,0]
        TN_c1 = cm[1,1]

        # Sens(C1)
        sens_c1 = TP_c1/(TP_c1+FN_c1) if (TP_c1+FN_c1)>0 else 0
        folds_sens_c1.append(sens_c1)
        folds_sens_c1_n.append(TP_c1+FN_c1)
        folds_sens_c1_x.append(TP_c1)

        # Spec(C1)
        spec_c1 = TN_c1/(TN_c1+FP_c1) if (TN_c1+FP_c1)>0 else 0
        folds_spec_c1.append(spec_c1)
        folds_spec_c1_n.append(TN_c1+FP_c1)
        folds_spec_c1_x.append(TN_c1)

        # PPV(C1)
        ppv_c1 = TP_c1/(TP_c1+FP_c1) if (TP_c1+FP_c1)>0 else 0
        folds_ppv_c1.append(ppv_c1)
        folds_ppv_c1_n.append(TP_c1+FP_c1)
        folds_ppv_c1_x.append(TP_c1)

        # NPV(C1)
        npv_c1 = TN_c1/(TN_c1+FN_c1) if (TN_c1+FN_c1)>0 else 0
        folds_npv_c1.append(npv_c1)
        folds_npv_c1_n.append(TN_c1+FN_c1)
        folds_npv_c1_x.append(TN_c1)

        # F1(C1)
        f1_c1 = 0
        if ppv_c1+sens_c1>0:
            f1_c1 = 2*ppv_c1*sens_c1/(ppv_c1+sens_c1)
        folds_f1_c1.append(f1_c1)
        folds_f1_c1_n.append(total_samples)

        # ROC 曲线（以C2为例）
        fpr_c2, tpr_c2, _ = roc_curve(all_labels, all_probs[:,1], pos_label=1)
        fpr_c2, tpr_c2 = ensure_roc_start_end(fpr_c2, tpr_c2)
        fpr_c2, tpr_c2 = remove_consecutive_duplicates(fpr_c2, tpr_c2)
        auc_c2_val = auc(fpr_c2, tpr_c2)
        folds_auc_c2.append(auc_c2_val)
        n_plus_C2 = sum(all_labels==1)
        n_minus_C2= sum(all_labels==0)
        folds_auc_c2_n.append(n_plus_C2*n_minus_C2)
        
        # 保存本折 ROC 数据用于后续汇总（新增加）
        folds_roc_data_C2.append((fpr_c2, tpr_c2, auc_c2_val))

        # 单独绘制本折 ROC 曲线（原逻辑）
        plt.figure(figsize=(8,6))
        plt.plot(fpr_c2, tpr_c2, 'b-', label=f'{label_name_C2}2 AUC={auc_c2_val:.3f}')
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        # 可选：让横纵比相同
        # plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"ROC({label_name_C2}) fold={i}",fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        # 设置图例字体大小
        plt.legend(loc="lower right", fontsize=14)
        plt.savefig(f"{save_dir}/{splitsLabelPath}_{featureModel}_{method_name}_ROC_{label_name_C2}_fold_{i}.png", dpi=400)
        plt.close()

        fpr_c1, tpr_c1, _ = roc_curve(all_labels, all_probs[:,0], pos_label=0)
        fpr_c1, tpr_c1 = ensure_roc_start_end(fpr_c1, tpr_c1)
        fpr_c1, tpr_c1 = remove_consecutive_duplicates(fpr_c1, tpr_c1)
        auc_c1_val = auc(fpr_c1, tpr_c1)
        folds_auc_c1.append(auc_c1_val)
        n_plus_C1 = sum(all_labels==0)
        n_minus_C1= sum(all_labels==1)
        folds_auc_c1_n.append(n_plus_C1*n_minus_C1)

        folds_roc_data_C1.append((fpr_c1, tpr_c1, auc_c1_val))
        plt.figure(figsize=(8,6))
        plt.plot(fpr_c1, tpr_c1, 'b-', label=f'{label_name_C1} AUC={auc_c1_val:.3f}')
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title(f"ROC({label_name_C1}) fold={i}",fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        # 设置图例字体大小
        plt.legend(loc="lower right", fontsize=14)
        plt.savefig(f"{save_dir}/{splitsLabelPath}_{featureModel}_{method_name}_ROC_{label_name_C1}_fold_{i}.png", dpi=400)
        plt.close()

    # 选出 best fold（基于 Youden 指数）
    if 'MSI' or 'TMB' in PROJECT_NAMES:
        folds_sens_c2_ = np.array(folds_sens_c2)
        best_idx = np.argmax(folds_sens_c2_)
        best_fold_case_results = fold_case_results[best_idx]
    else:
        folds_yi = np.array(folds_yi)
        best_idx = np.argmax(folds_yi)
        best_fold_case_results = fold_case_results[best_idx]

    # 构造 summary_dict（与原先类似）
    def mean_std_str(arr):
        return f"{np.mean(arr):.4f} ± {np.std(arr):.4f}"

    folds_acc   = np.array(folds_acc)
    folds_auc_c2= np.array(folds_auc_c2)
    folds_sens_c2= np.array(folds_sens_c2)
    folds_spec_c2= np.array(folds_spec_c2)
    folds_ppv_c2 = np.array(folds_ppv_c2)
    folds_npv_c2 = np.array(folds_npv_c2)
    folds_f1_c2  = np.array(folds_f1_c2)
    folds_auc_c1 = np.array(folds_auc_c1)
    folds_sens_c1= np.array(folds_sens_c1)
    folds_spec_c1= np.array(folds_spec_c1)
    folds_ppv_c1 = np.array(folds_ppv_c1)
    folds_npv_c1 = np.array(folds_npv_c1)
    folds_f1_c1  = np.array(folds_f1_c1)

    summary_dict = {
        'best_idx': best_idx,
        'best_fold_name': f"fold_{best_idx}",
        'Feature extraction': featureModel,
        'best_fold_case_results': best_fold_case_results,  # 逐例结果
        f'ROC({label_name_C2})': folds_roc_data_C2[best_idx],  # 新增：最佳折的ROC数据 (fpr, tpr, auc)
        f'ROC({label_name_C1})':folds_roc_data_C1[best_idx], 
        'Youden index': {
            'best_fold': folds_yi[best_idx],
            'mean_std': mean_std_str(folds_yi),
        },
        f'AUC({label_name_C2})': {
            'best_fold': folds_auc_c2[best_idx],
            'mean_std': mean_std_str(folds_auc_c2),
            'auc_n': np.array(folds_auc_c2_n)
        },
        'Accuracy': {
            'best_fold': folds_acc[best_idx],
            'mean_std': mean_std_str(folds_acc),
            'n_array': np.array(folds_acc_n),
            'x_array': np.array(folds_acc_x),
        },
        f'Sensitivity({label_name_C2})': {
            'best_fold': folds_sens_c2[best_idx],
            'mean_std': mean_std_str(folds_sens_c2),
            'n_array': np.array(folds_sens_c2_n),
            'x_array': np.array(folds_sens_c2_x),
        },
        f'Specificity({label_name_C2})': {
            'best_fold': folds_spec_c2[best_idx],
            'mean_std': mean_std_str(folds_spec_c2),
            'n_array': np.array(folds_spec_c2_n),
            'x_array': np.array(folds_spec_c2_x),
        },
        f'PPV({label_name_C2})': {
            'best_fold': folds_ppv_c2[best_idx],
            'mean_std': mean_std_str(folds_ppv_c2),
            'n_array': np.array(folds_ppv_c2_n),
            'x_array': np.array(folds_ppv_c2_x),
        },
        f'NPV({label_name_C2})': {
            'best_fold': folds_npv_c2[best_idx],
            'mean_std': mean_std_str(folds_npv_c2),
            'n_array': np.array(folds_npv_c2_n),
            'x_array': np.array(folds_npv_c2_x),
        },
        f'F1({label_name_C2})': {
            'best_fold': folds_f1_c2[best_idx],
            'mean_std': mean_std_str(folds_f1_c2),
        },
        f'AUC({label_name_C1})': {
            'best_fold': folds_auc_c1[best_idx],
            'mean_std': mean_std_str(folds_auc_c1),
            'auc_n': np.array(folds_auc_c1_n),
        },
        f'Sensitivity({label_name_C1})': {
            'best_fold': folds_sens_c1[best_idx],
            'mean_std': mean_std_str(folds_sens_c1),
            'n_array': np.array(folds_sens_c1_n),
            'x_array': np.array(folds_sens_c1_x),
        },
        f'Specificity({label_name_C1})': {
            'best_fold': folds_spec_c1[best_idx],
            'mean_std': mean_std_str(folds_spec_c1),
            'n_array': np.array(folds_spec_c1_n),
            'x_array': np.array(folds_spec_c1_x),
        },
        f'PPV({label_name_C1})': {
            'best_fold': folds_ppv_c1[best_idx],
            'mean_std': mean_std_str(folds_ppv_c1),
            'n_array': np.array(folds_ppv_c1_n),
            'x_array': np.array(folds_ppv_c1_x),
        },
        f'NPV({label_name_C1})': {
            'best_fold': folds_npv_c1[best_idx],
            'mean_std': mean_std_str(folds_npv_c1),
            'n_array': np.array(folds_npv_c1_n),
            'x_array': np.array(folds_npv_c1_x),
        },
        f'F1({label_name_C1})': {
            'best_fold': folds_f1_c1[best_idx],
            'mean_std': mean_std_str(folds_f1_c1),
        },
    }

    return summary_dict

def mcnemar_acc_p(baseline_cases, compare_cases):
    """
    baseline_cases: [(id, true_label, pred_label), ...]
    compare_cases:  [(id, true_label, pred_label), ...]
    返回: p-value (float)
    """
    # 将两个模型的结果按样本id建立字典，方便匹配
    baseline_dict = {case[0]: case for case in baseline_cases}
    compare_dict = {case[0]: case for case in compare_cases}
    
    # 记录discordant pairs：
    # b: baseline预测正确，compare预测错误的样本数
    # c: baseline预测错误，compare预测正确的样本数
    b = 0  
    c = 0  
    
    # 遍历所有样本（以baseline为主）
    for sample_id, baseline_case in baseline_dict.items():
        if sample_id not in compare_dict:
            continue  # 如果compare中没有对应样本，则跳过
        _, true_label, baseline_pred = baseline_case
        _, _, compare_pred = compare_dict[sample_id]
        
        baseline_correct = (baseline_pred == true_label)
        compare_correct = (compare_pred == true_label)
        
        if baseline_correct and not compare_correct:
            b += 1
        elif not baseline_correct and compare_correct:
            c += 1
    
    # 如果没有discordant pairs，则无法计算差异，此时返回p值为1.0
    if (b + c) == 0:
        return 1.0
    
    # 使用带连续性校正的McNemar检验统计量
    chi_square = ((abs(b - c) - 1) ** 2) / (b + c)
    
    # 自由度为1，根据卡方分布计算p值
    p_value = 1 - chi2.cdf(chi_square, df=1)
    
    return p_value



# def independent_chi2_2x2(a, b, c, d):
#     """
#     独立卡方：2x2表
#     [ [a, b],
#       [c, d] ]
#     """
#     table = np.array([[a, b],[c, d]])
#     chi2, p, dof, ex = chi2_contingency(table, correction=False)
#     return p

def independent_chi2_2x2(b_tp, b_fp, m_tp, m_fp):
    # 构造2x2列联表
    table = [[b_tp, b_fp],
             [m_tp, m_fp]]
    # 执行 Fisher 精确检验
    odds_ratio, p_value = fisher_exact(table)
    return p_value

    
def compute_p_values(RESULT_TABLE, ALL_BEST_FOLD_CASES):
    """
    对同一Process下, 以 (Model=uni, Method=clam) 为基准,
    针对 Accuracy, Sensitivity/Specificity(C2/C1) 做 McNemar，
    针对 PPV/NPV(C2/C1) 做独立卡方。
    写入 (metric, 'P-value')。
    """
    processes = RESULT_TABLE.index.levels[0]  # Angiogenesis, Autophagy, Stemness
    for proc in processes:
        # 基准: (proc, 'uni', 'clam')
        baseline_key = (proc, 'uni', 'clam')
        if baseline_key not in ALL_BEST_FOLD_CASES:
            continue
        baseline_cases = ALL_BEST_FOLD_CASES[baseline_key]

        # 同一process下的所有 (model, method)
        # RESULT_TABLE.index 是 multiIndex, 过滤 process=proc
        subidx = [idx for idx in ALL_BEST_FOLD_CASES.keys() if idx[0]==proc]

        for row_key in subidx:
            if row_key == baseline_key:
                # 不和自己比
                continue
            compare_cases = ALL_BEST_FOLD_CASES[row_key]

            # ---- 1) Accuracy => McNemar
            # 这里把Accuracy当成“是否判对”
            p_mcnemar_acc = mcnemar_acc_p(baseline_cases, compare_cases)
            RESULT_TABLE.loc[row_key, ('Accuracy','P-value')] = f"{p_mcnemar_acc:.3g}"

            # ---- 2) Sens(C2) => McNemar
            #   sens(C2)是“C2的对错”，所以只考虑 label=1 的样本 => 对 vs. 错
            #   需要先过滤 label=1
            #   baseline_cases 中 label=1 => baseline_correct?
            #   compare_cases 同理
            #   只对 label=1 的子集做 McNemar
            #   我们可以写一个辅助函数
            p_sens_c2 = mcnemar_sensitivity(baseline_cases, compare_cases, pos_label=1)
            RESULT_TABLE.loc[row_key, (f'Sensitivity({label_name_C2})','P-value')] = f"{p_sens_c2:.3g}"

            # ---- 3) Spec(C2) => McNemar
            #   只对 label=0 的样本 => baseline_correct? compare_correct?
            p_spec_c2 = mcnemar_sensitivity(baseline_cases, compare_cases, pos_label=0)
            RESULT_TABLE.loc[row_key, (f'Specificity({label_name_C2})','P-value')] = f"{p_spec_c2:.3g}"

            # ---- 4) Sens(C1) => label=0当正类
            #   只对 label=0 => baseline_correct? compare_correct?
            #   “Sensitivity(C1)” 相当于“TP/(TP+FN)”当 label=0 为“阳性”
            p_sens_c1 = mcnemar_sensitivity(baseline_cases, compare_cases, pos_label=0)
            RESULT_TABLE.loc[row_key, (f'Sensitivity({label_name_C1})','P-value')] = f"{p_sens_c1:.3g}"

            # ---- 5) Spec(C1) => label=1
            p_spec_c1 = mcnemar_sensitivity(baseline_cases, compare_cases, pos_label=1)
            RESULT_TABLE.loc[row_key, (f'Specificity({label_name_C1})','P-value')] = f"{p_spec_c1:.3g}"

            # ---- 6) PPV(C2) => 独立卡方
            #   PPV(C2) => row=[TP, FP], baseline vs. compare
            #   先数 baseline_TP, baseline_FP, compare_TP, compare_FP
            b_tp, b_fp = count_tp_fp(baseline_cases, pos_label=1)
            m_tp, m_fp = count_tp_fp(compare_cases,  pos_label=1)
            p_ppv_c2 = independent_chi2_2x2(b_tp, b_fp, m_tp, m_fp)
            RESULT_TABLE.loc[row_key, (f'PPV({label_name_C2})','P-value')] = f"{p_ppv_c2:.3g}"

            # ---- 7) NPV(C2) => row=[TN, FN]
            b_tn, b_fn = count_tn_fn(baseline_cases, pos_label=1)
            m_tn, m_fn = count_tn_fn(compare_cases,  pos_label=1)
            p_npv_c2 = independent_chi2_2x2(b_tn, b_fn, m_tn, m_fn)
            RESULT_TABLE.loc[row_key, (f'NPV({label_name_C2})','P-value')] = f"{p_npv_c2:.3g}"

            # ---- 8) PPV(C1) => pos_label=0
            b_tp_c1, b_fp_c1 = count_tp_fp(baseline_cases, pos_label=0)
            m_tp_c1, m_fp_c1 = count_tp_fp(compare_cases,  pos_label=0)
            p_ppv_c1 = independent_chi2_2x2(b_tp_c1, b_fp_c1, m_tp_c1, m_fp_c1)
            RESULT_TABLE.loc[row_key, (f'PPV({label_name_C1})','P-value')] = f"{p_ppv_c1:.3g}"

            # ---- 9) NPV(C1) => row=[TN, FN], pos_label=0
            b_tn_c1, b_fn_c1 = count_tn_fn(baseline_cases, pos_label=0)
            m_tn_c1, m_fn_c1 = count_tn_fn(compare_cases,  pos_label=0)
            p_npv_c1 = independent_chi2_2x2(b_tn_c1, b_fn_c1, m_tn_c1, m_fn_c1)
            RESULT_TABLE.loc[row_key, (f'NPV({label_name_C1})','P-value')] = f"{p_npv_c1:.3g}"

    return RESULT_TABLE

# 辅助函数：在 label=pos_label 的子集中做 McNemar
def mcnemar_sensitivity(baseline_cases, compare_cases, pos_label=1):
    """
    仅统计 label=pos_label 的样本, baseline vs. compare 是否判对
    """
    base_dict = {item[0]: (item[1], item[2]) for item in baseline_cases}
    n_plus_minus = 0
    n_minus_plus = 0
    for (cid, lab, pred) in compare_cases:
        if cid not in base_dict:
            continue
        base_lab, base_pred = base_dict[cid]
        assert base_lab == lab,'金标准不对称'
        
        if lab==pos_label:
            # baseline_correct?
            b_corr = (base_pred==lab)
            c_corr = (pred==lab)
            if b_corr and (not c_corr):
                n_plus_minus+=1
            elif (not b_corr) and c_corr:
                n_minus_plus+=1
    if (n_plus_minus + n_minus_plus)==0:
        return 1.0
    chi2_stat = (abs(n_plus_minus - n_minus_plus)-1)**2 / (n_plus_minus + n_minus_plus)
    p_val = 1 - chi2.cdf(chi2_stat, df=1)
    return p_val

# 统计 TP/FP: pos_label=1 时, TP=预测=1且真实=1
def count_tp_fp(case_list, pos_label=1):
    tp=0
    fp=0
    for (cid, lab, pred) in case_list:
        if pred==pos_label:
            if lab==pos_label:
                tp+=1
            else:
                fp+=1
    return tp, fp

# 统计 TN/FN: pos_label=1 时, TN=预测=0且真实=0
def count_tn_fn(case_list, pos_label=1):
    tn=0
    fn=0
    for (cid, lab, pred) in case_list:
        if pred!=pos_label:
            # 预测为非pos_label
            if lab!=pos_label:
                tn+=1
            else:
                fn+=1
    return tn, fn


DATASETS_ID = {'train': 0, 'val': 1, 'test': 2, 'all': -1}
INDEPENDENT_TEST = True
save_dir = 'eval_result_TMB'
os.makedirs(f'{save_dir}', exist_ok=True)
MULTI_INSTANCE_MODELS = ['clam','mil']
FEATURE_MODELS = ['uni','CTrans','resnet']
# PROJECT_NAMES = ['Angiogenesis', 'Autophagy', 'Stemness']
PROJECT_NAMES = ['TMB']
# PROJECT_NAMES = ['MSI']
# PROJECT_NAMES = ['Autophagy']

label_name_C1 = 'TMB-L'
label_name_C2 = 'TMB-H'

row_index = pd.MultiIndex.from_product(
    [PROJECT_NAMES, FEATURE_MODELS, MULTI_INSTANCE_MODELS],
    names=['Process', 'Model', 'Method']
)

metrics = [
    f'AUC({label_name_C2})', 'Accuracy', f'Sensitivity({label_name_C2})', f'Specificity({label_name_C2})', 
    f'PPV({label_name_C2})', f'NPV({label_name_C2})', 'Youden index', f'F1({label_name_C2})',
    f'AUC({label_name_C1})', f'Sensitivity({label_name_C1})', f'Specificity({label_name_C1})', 
    f'PPV({label_name_C1})', f'NPV({label_name_C1})', f'F1({label_name_C1})'
]
subcols = ['Best fold', 'Mean ± SD']
# 加一个 P-value 列（后面要写 p 值）
# 但是有些指标(如 AUC, F1, YI)并不需要 p 值；可选加也行
all_subcols = ['Best fold','Mean ± SD','P-value']

metric_cols = pd.MultiIndex.from_product(
    [metrics, all_subcols],
    names=['Metric', 'Details']
)

feat_col = pd.MultiIndex.from_tuples(
    [
       ('Feature extraction', ''), 
       ('Best fold name', '')
    ],
    names=['Metric', 'Details']
)
full_col_index = feat_col.append(metric_cols)
RESULT_TABLE = pd.DataFrame(index=row_index, columns=full_col_index)

BASE_PATH = 'best_result'



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

fiveNamesDict = {
    'fold_0':'s_0_checkpoint.pt',
    'fold_1':'s_1_checkpoint.pt',
    'fold_2':'s_2_checkpoint.pt',
    'fold_3':'s_3_checkpoint.pt',
    'fold_4':'s_4_checkpoint.pt',
}
ALL_BEST_FOLD_CASES = {} 
ALL_BEST_ROC_C1 = {} 
ALL_BEST_ROC_C2 = {} 

for method_name in MULTI_INSTANCE_MODELS:  # clam/mil
    for process_name in PROJECT_NAMES:  # MSI/TMB等任务
        for featureModel in FEATURE_MODELS:  # uni/CTrans/resnet
            project_path = os.path.join(BASE_PATH, method_name, process_name)
            featurn_models = os.path.join(os.path.join(project_path, featureModel), os.listdir(os.path.join(project_path, featureModel))[0])
            originLabelPath = originLabelDict[process_name]
            dataPath = os.path.join('RESULTS_DIRECTORY', featureModel)

            # 构建 dataset
            EvalDataset = Generic_MIL_Dataset_PP(
                csv_path=originLabelPath,
                data_dir=dataPath,
                shuffle=False, 
                print_info=True,
                label_dict={'C1':0, 'C2':1},
                patient_strat=False,
                ignore=[]
            )

            # 调用 evalClamModel 进行评估
            summary_dict = evalClamModel(
                clamPath=featurn_models,
                EvalDataset=EvalDataset,
                splitsLabelPath=process_name,
                featureModel=featureModel,
                process_name=process_name,
                method_name=method_name
            )

            row_key = (process_name, featureModel, method_name)
            RESULT_TABLE.loc[row_key, ('Feature extraction','')] = summary_dict['Feature extraction']
            RESULT_TABLE.loc[row_key, ('Best fold name','')] = summary_dict['best_fold_name']

            # 复制最优折模型（原逻辑保持不变）
            best_idx = summary_dict['best_idx']
            best_fold_ckpt = fiveNamesDict[summary_dict['best_fold_name']]
            os.makedirs(featurn_models.replace('best_result','best_model_one'), exist_ok=True)
            bestModelPath_src = os.path.join(featurn_models, best_fold_ckpt)
            bestconfigPath_src= os.path.join(featurn_models, 'experiment_task_1_tumor_vs_normal.txt')
            shutil.copy(bestModelPath_src, bestModelPath_src.replace('best_result','best_model_one'))
            shutil.copy(bestconfigPath_src, bestconfigPath_src.replace('best_result','best_model_one'))

            # 将需要的指标写入表格（原逻辑不变）……
            for metric in metrics:
                if metric not in summary_dict:
                    continue
                p_val = summary_dict[metric]['best_fold']
                mean_std_val = summary_dict[metric]['mean_std']
                if 'n_array' in summary_dict[metric]:
                    n_arr = summary_dict[metric]['n_array']
                    x_arr = summary_dict[metric]['x_array']
                    n_val = n_arr[best_idx]
                    x_val = x_arr[best_idx]
                    if n_val > 0:
                        l, u = proportion_confidence_interval(p_val, n_val)
                        best_fold_str = (f"{p_val*100:.2f}% "
                                         f"({x_val}/{n_val}, "
                                         f"{l*100:.2f}-{u*100:.2f}%)")
                    else:
                        best_fold_str = f"{p_val*100:.2f}% (N=0)"
                else:
                    best_fold_str = f"{p_val*100:.2f}%"
                ms = mean_std_val.split('±')
                m = float(ms[0].strip())
                s = float(ms[1].strip())
                mean_std_str = f"{m*100:.2f}% ± {s*100:.2f}%"

                RESULT_TABLE.loc[row_key, (metric, 'Best fold')] = best_fold_str
                RESULT_TABLE.loc[row_key, (metric, 'Mean ± SD')] = mean_std_str
                RESULT_TABLE.loc[row_key, (metric, 'P-value')] = ""

            # 保存最佳折的 case-level 结果到全局字典
            ALL_BEST_FOLD_CASES[row_key] = summary_dict['best_fold_case_results']
            # 新增：记录该组合的最佳ROC数据
            ALL_BEST_ROC_C1[row_key] = summary_dict[f'ROC({label_name_C1})']
            ALL_BEST_ROC_C2[row_key] = summary_dict[f'ROC({label_name_C2})']

StandardName = {
    'uni':'UNI',
    'clam':'CLAM',
    'CTrans':'CTrans',
    'resnet':'ResNet50',
    'mil':'Classic MIL'
}

for project_name in PROJECT_NAMES:
    # 筛选出属于当前任务的所有组合
    relevant_keys = [key for key in ALL_BEST_ROC_C2.keys() if key[0] == project_name]
    if not relevant_keys:
        print(f"任务 {project_name} 没有对应的ROC数据，跳过绘图。")
        continue

    plt.figure(figsize=(8,6), dpi=400)
    for key in relevant_keys:
        # key = (process_name, featureModel, method_name)
        fpr, tpr, auc_val = ALL_BEST_ROC_C2[key]
        label_str = f"{StandardName[key[1]]}+{StandardName[key[2]]} (AUC={auc_val:.3f})"  # 例如：uni_clam (AUC=0.850)
        plt.plot(fpr, tpr, label=label_str)  # 默认自动分配不同颜色

    # 绘制对角线
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    # 可选：让横纵比相同
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("False Positive Rate",fontsize=14)
    plt.ylabel("True Positive Rate",fontsize=14)
    plt.title(f"{project_name} Best ROC",fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="lower right",fontsize=10)
    # 保存图像，文件名以任务名称命名
    plt.savefig(f"{save_dir}/{project_name}_combined_ROC_{label_name_C2}.png", bbox_inches='tight', dpi=400)
    plt.close()

for project_name in PROJECT_NAMES:
    # 筛选出属于当前任务的所有组合
    relevant_keys = [key for key in ALL_BEST_ROC_C2.keys() if key[0] == project_name]
    if not relevant_keys:
        print(f"任务 {project_name} 没有对应的ROC数据，跳过绘图。")
        continue

    plt.figure(figsize=(8,6), dpi=400)
    for key in relevant_keys:
        # key = (process_name, featureModel, method_name)
        fpr, tpr, auc_val = ALL_BEST_ROC_C1[key]
        label_str = f"{StandardName[key[1]]}+{StandardName[key[2]]} (AUC={auc_val:.3f})"  # 例如：uni_clam (AUC=0.850)
        plt.plot(fpr, tpr, label=label_str)  # 默认自动分配不同颜色

    # 绘制对角线
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    # 可选：让横纵比相同
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("False Positive Rate",fontsize=14)
    plt.ylabel("True Positive Rate",fontsize=14)
    plt.title(f"{project_name} Best ROC",fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc="lower right",fontsize=10)
    # 保存图像，文件名以任务名称命名
    plt.savefig(f"{save_dir}/{project_name}_combined_ROC_{label_name_C1}.png", bbox_inches='tight', dpi=400)
    plt.close()

for project_name in PROJECT_NAMES:
    # 1) 找出只属于该功能的 (功能, 模型, 方式) 键
    relevant_keys = [
        (func, model, approach)
        for (func, model, approach) in ALL_BEST_FOLD_CASES.keys()
        if func == project_name
    ]
    # 如果该功能在字典中根本没有出现，则跳过
    if not relevant_keys:
        print(f"功能 {project_name} 在 ALL_BEST_FOLD_CASES 中没有数据，跳过。")
        continue
    
    # 2) 收集所有文件名(行索引)
    filenames = set()
    for (func, model, approach) in relevant_keys:
        triple_list = ALL_BEST_FOLD_CASES[(func, model, approach)]
        for (filename, gold, pred) in triple_list:
            filenames.add(filename)
    filenames = sorted(filenames)

    # 3) 为该功能构建多级列索引：模型 -> 方式 -> (文件名/金标准/预测结果)
    columns_tuples = []
    for (func, model, approach) in relevant_keys:
        # 每个 (模型, 方式) 下有 3 个子列
        columns_tuples.append((model, approach, '文件名'))
        columns_tuples.append((model, approach, '金标准'))
        columns_tuples.append((model, approach, '预测结果'))

    # 多级列：最外层是 模型，第二层是 方式，第三层是 (文件名/金标准/预测结果)
    multi_index = pd.MultiIndex.from_tuples(
        columns_tuples,
        names=["模型", "方式", None]
    )

    # 4) 创建空的 DataFrame，并填充数据
    df_sub = pd.DataFrame(index=filenames, columns=multi_index)
    for (func, model, approach) in relevant_keys:
        triple_list = ALL_BEST_FOLD_CASES[(func, model, approach)]
        for (filename, gold, pred) in triple_list:
            df_sub.loc[filename, (model, approach, '文件名')] = filename
            df_sub.loc[filename, (model, approach, '金标准')] = gold
            df_sub.loc[filename, (model, approach, '预测结果')] = pred

    # 5) 输出到 Excel
    output_file = os.path.join(save_dir, f"{project_name}.xlsx")
    df_sub.to_excel(output_file, merge_cells=True)




# 现在执行 compute_p_values 并保存
RESULT_TABLE = compute_p_values(RESULT_TABLE, ALL_BEST_FOLD_CASES)
RESULT_TABLE.to_excel(f"{save_dir}/final_eval_table_with_pvalue.xlsx")
print("Done. Check final_eval_table_with_pvalue.xlsx")

