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


BASE_PATH = 'result/prognosis/uni/1'
DEVICE = 'cuda:0'
# MULTI_INSTANCE_MODELS = ['clam','mil']
# PROJECT_NAMES = ['Angiogenesis', 'Autophagy','Stemness']
# FEATURE_MODELS = ['CTrans','uni','resnet']
MULTI_INSTANCE_MODELS = 'clam'
PROJECT_NAMES = 'prognosis'
FEATURE_MODELS = 'uni'
EXTERNAL_DATA_PATH = 'RESULTS_DIRECTORY/uni/pt_files' #
# EXTERNAL_DATA_PATH = 'ExternalDataCrc/RESULTS_DIRECTORY/data2/***/pt_files' #
EMBED_DIM = {
                'uni': 1024,
                'CTrans':768,
                'resnet':1024    
            }
temp_names = [
"TCGA-AA-3858-01Z-00-DX1.6336815f-9887-4f74-a15d-78e7f6cacb59_20x",
"TCGA-AA-A01Z-01Z-00-DX1.9724B55C-C5D9-4C8B-AA05-76C21BA1F046_20x",
"TCGA-AA-3986-01Z-00-DX1.db60e495-c0eb-416c-b65b-55ce62ed10b0_20x",
"TCGA-AA-A029-01Z-00-DX1.36BA3129-431D-4AE5-98E6-BA064D0B5062_20x",
"TCGA-A6-3810-01Z-00-DX1.2940ca70-013a-4bc3-ad6a-cf4d9ffa77ce_20x",
"TCGA-AA-3980-01Z-00-DX1.93383cb9-59a7-431d-b268-3c3d59a1120e_20x",
"TCGA-D5-6931-01Z-00-DX1.c1d00654-b5ff-4485-90a6-97ae9e7bd7fa_20x",
"TCGA-AA-A02J-01Z-00-DX1.1326204B-9264-482C-9F75-795DD085C0DF_20x",
"TCGA-AA-3549-01Z-00-DX1.2fe99d54-c61b-4867-bafe-efe4f291c429_20x",
"TCGA-A6-6653-01Z-00-DX1.e130666d-2681-4382-9e7a-4a4d27cb77a4_20x",
"TCGA-A6-2678-01Z-00-DX1.bded5c5c-555a-492a-91c7-151492d0ee5e_20x",
"TCGA-AA-3854-01Z-00-DX1.1564d865-6653-4be1-951e-ea9fab0102a7_20x",
"TCGA-CM-5864-01Z-00-DX1.2cb87875-6cae-4d8e-9c93-4a83941c0ca9_20x",
"TCGA-CM-5341-01Z-00-DX1.af4f75ff-3971-4639-8ef4-918ef4b29df0_20x",
"TCGA-A6-2674-01Z-00-DX1.d301f1f5-6f4a-49e6-9c93-f4e8b7f616b8_20x",
"TCGA-AA-3979-01Z-00-DX1.e63b4db2-dc9b-4afb-a288-89a905beacd0_20x",
"TCGA-G4-6320-01Z-00-DX1.09f11d38-4d47-44c9-b8d6-4d4910c6280e_20x",
"TCGA-AA-3856-01Z-00-DX1.973974e7-fcfe-4866-bc0c-50645c6c304b_20x",
"TCGA-D5-6539-01Z-00-DX1.fe2a2e60-1db0-4019-9920-99416b34f05e_20x",
"TCGA-AA-3970-01Z-00-DX1.712c069a-aeaf-498b-80fa-7bb481b13825_20x",
"TCGA-AA-A00R-01Z-00-DX1.7520405C-E7DD-46A4-BB68-ADFA511AEA64_20x",
"TCGA-G4-6321-01Z-00-DX1.20bd4687-4b24-4666-a722-d42b9731136e_20x",
"TCGA-AA-3552-01Z-00-DX1.84133d42-9a39-44b5-a1ec-a5382650c939_20x",
"TCGA-CM-6171-01Z-00-DX1.74d4391e-3dbc-4ad4-b188-3b11ac65e6d8_20x",
"TCGA-CK-5912-01Z-00-DX1.23a955f3-a1ed-4cb3-8e49-cbb3f789f3f5_20x",
"TCGA-CA-5796-01Z-00-DX1.88141789-4240-4ab7-8db1-e4cb7ee1ebda_20x",
"TCGA-G4-6295-01Z-00-DX1.9e7ae22f-daac-42cb-a879-bcf505d1c725_20x",
"TCGA-AA-3664-01Z-00-DX1.bd07e7ef-0acb-43d8-a4f6-15b3442d2ed5_20x",
"TCGA-AA-3519-01Z-00-DX1.82e03504-31d8-43d5-8d3f-01d9016af0fe_20x",
"TCGA-A6-2680-01Z-00-DX1.7b77c0fb-f51d-4d16-ae77-f7615b1d0b87_20x",
"TCGA-DM-A1D9-01Z-00-DX1.C286F663-142A-4F8E-BFCD-56E33F73F7E8_20x",
"TCGA-CM-6168-01Z-00-DX1.96af6eb2-9d51-4671-baf8-1a73d0c66869_20x",
"TCGA-G4-6314-01Z-00-DX1.bea21980-9584-4382-9de3-4c5114edb10d_20x",
"TCGA-G4-6311-01Z-00-DX1.f1b98598-dbd8-4ba5-9ec7-5c93ccc82c81_20x",
"TCGA-AA-3526-01Z-00-DX1.82876320-2866-4ffa-81d7-3278f7150fc3_20x",
"TCGA-CM-6161-01Z-00-DX1.552104aa-6fd7-4d53-918b-fe67d359815c_20x",
"TCGA-F4-6807-01Z-00-DX1.84bfb631-af3d-45e7-a7db-730844a53625_20x",
"TCGA-CM-5349-01Z-00-DX1.d893eb9a-0321-4052-acfc-8c9a6e463921_20x",
"TCGA-F4-6809-01Z-00-DX1.5ab8333f-0c77-4685-8701-4130a93e6f3a_20x",
"TCGA-AA-A02K-01Z-00-DX1.732DD8F9-A21A-4E97-A779-3400A6C3D19D_20x",
"TCGA-AA-3673-01Z-00-DX1.a80676fa-5481-4b63-9639-dbeb31ae82d8_20x",
"TCGA-F4-6463-01Z-00-DX1.a3fa6fb4-ce9d-4f0d-b5f7-3c9da7322cd0_20x",
"TCGA-A6-5667-01Z-00-DX1.1973b80d-b6b8-4ed8-9bc1-3aef51fbd9e6_20x",
"TCGA-AZ-6598-01Z-00-DX1.1fc4cd61-4524-413b-b36d-ad438785bc06_20x",
"TCGA-AA-3812-01Z-00-DX1.c501fc71-8370-4034-b32a-1bb7cd846881_20x",
"TCGA-AZ-6600-01Z-00-DX1.9afe2f8f-bcfe-43df-a83b-6c183f226757_20x",
"TCGA-CM-6169-01Z-00-DX1.0381c243-02b8-4f1d-840c-19ef44d4b92c_20x",
"TCGA-D5-7000-01Z-00-DX1.fb08c430-2c8c-486b-a39d-7d28c5eae189_20x",
"TCGA-WS-AB45-01Z-00-DX1.1FD99E7A-830F-40DC-98CD-53C62C678AC6_20x",
"TCGA-SS-A7HO-01Z-00-DX1.D20B9109-F984-40DE-A4F1-2DFC61002862_20x",
"TCGA-DM-A285-01Z-00-DX1.219e2829-8ffd-4b51-adce-cfd48293191b_20x",
"TCGA-CA-6718-01Z-00-DX1.9774472f-a29a-4b2b-8e50-ccbf9e5f9686_20x",
"TCGA-F4-6460-01Z-00-DX1.92a182ea-f22a-4d74-bfb6-34d3cd757dce_20x",
"TCGA-AA-3994-01Z-00-DX1.ca18c0cb-88b4-4a31-be1f-cca57dfadabc_20x",
"TCGA-DM-A28M-01Z-00-DX1.055b2d62-8a1e-4bdf-a49e-123ad0de657b_20x",
"TCGA-AD-6901-01Z-00-DX1.0a69c0b5-6238-4c1a-bbbd-ea743bf6fc98_20x",
"TCGA-A6-2676-01Z-00-DX1.c465f6e0-b47c-48e9-bdb1-67077bb16c67_20x",
"TCGA-D5-5537-01Z-00-DX1.14709d4c-eba0-48d0-87b8-5f34f74429d6_20x",
"TCGA-NH-A8F8-01Z-00-DX1.0C13D583-0BCE-44F7-A4E6-5994FE97B99C_20x",
"TCGA-A6-3809-01Z-00-DX1.c26f03e8-c285-4a66-925d-ae9cba17d7b3_20x",
"TCGA-AA-3681-01Z-00-DX1.576342cf-0f40-404a-b3c5-b33103f86777_20x",
"TCGA-AA-3531-01Z-00-DX1.19cdaa4b-5a53-4198-90da-5800827d90bf_20x",
"TCGA-AA-3696-01Z-00-DX1.947f2c09-dfe9-4fdb-bf1a-9bf46d67f617_20x",
"TCGA-AA-3977-01Z-00-DX1.08ffa326-08fd-4215-9bf7-81fcf33b4f5a_20x",
"TCGA-AA-3543-01Z-00-DX1.20129c52-157d-4d66-809f-d21694683c8d_20x",
"TCGA-AA-3517-01Z-00-DX1.dac0f9a3-fa10-42e7-acaf-e86fff0829d2_20x",
"TCGA-AA-3950-01Z-00-DX1.2a81cf11-4c16-4e9e-8809-6f63152060da_20x",
"TCGA-NH-A50T-01Z-00-DX1.4624B690-C0DE-42BD-852C-6EBABF40255F_20x",
"TCGA-CM-6172-01Z-00-DX1.a5d23c88-a173-46a2-b8dd-6d873b8216c7_20x",
"TCGA-AA-A024-01Z-00-DX1.5F24A31C-2F11-4768-9906-7BAB578C742D_20x",
"TCGA-AZ-5407-01Z-00-DX1.5218a617-9817-44f4-8f00-8e9e3d04bd70_20x",
"TCGA-AA-3524-01Z-00-DX1.b1aae264-87be-4514-8f9d-25660b39caa7_20x",
"TCGA-AA-3548-01Z-00-DX1.41949ab5-79f2-4729-9d54-c0fca1daf124_20x",
"TCGA-AZ-4315-01Z-00-DX1.1a2c2771-3e59-47c3-b380-42110c545e6b_20x",
"TCGA-G4-6306-01Z-00-DX1.962227ca-b0d6-4cf4-afea-8f7c2f9b2477_20x",
"TCGA-G4-6293-01Z-00-DX1.62ed5ed9-a79a-487a-bd6f-1f3f0571d44d_20x",
"TCGA-AA-3864-01Z-00-DX1.f6992bc7-ba05-4c30-9500-8f7b07b30f9a_20x",
"TCGA-AA-3842-01Z-00-DX1.8bbbd702-2b17-4c3e-a8bd-55c3ae8aaba3_20x",
"TCGA-AA-3680-01Z-00-DX1.9eef1b8f-c3c1-486f-83e7-a88182ce892a_20x",
"TCGA-CM-4752-01Z-00-DX1.ac26d5ca-f554-4766-a4c3-f90a8c327dd4_20x",
"TCGA-AA-A01G-01Z-00-DX1.8A288E53-BA38-4BAC-81B5-2E0E41EA0D85_20x",
"TCGA-AA-3556-01Z-00-DX1.63a74b91-44e8-4ffd-8737-bcf6992183c3_20x",
"TCGA-AA-A02H-01Z-00-DX1.5343879F-6C5D-48B3-8D78-D895ED118F42_20x",
"TCGA-A6-6648-01Z-00-DX1.88b9a490-0bed-43f3-bd74-1bf2810f6884_20x",
"TCGA-AA-3544-01Z-00-DX1.96850cbf-2305-4b65-8f06-db801af51cc3_20x",
"TCGA-AA-A00U-01Z-00-DX1.E83A6B38-D472-482F-89D7-FF61FB589371_20x",
"TCGA-QL-A97D-01Z-00-DX1.6B48E95D-BE3C-4448-A1AF-6988C00B7AF1_20x",
"TCGA-AA-3968-01Z-00-DX1.54b76478-a822-49b5-8286-dcbbb2fba2f8_20x",
"TCGA-AA-3561-01Z-00-DX1.1b5a2925-53f9-470f-a62c-cc2e5d5abb58_20x",
"TCGA-G4-6625-01Z-00-DX1.0fa26667-2581-4f96-a891-d78dbc3299b4_20x",
"TCGA-AZ-5403-01Z-00-DX1.1c557fea-6627-48e9-abb9-79da22c40cef_20x",
"TCGA-G4-6586-01Z-00-DX1.f19ef98f-9540-4b8d-bd13-5891e79b2576_20x",
"TCGA-CA-6715-01Z-00-DX1.d5db8085-f91a-4eee-b15f-61960af713af_20x",
"TCGA-CA-6717-01Z-00-DX1.08da75b7-a08f-46b3-a8c0-24f601ec4558_20x",
"TCGA-AZ-6601-01Z-00-DX1.40681471-3104-48be-8b57-55dba1f432f8_20x",
"TCGA-AD-6964-01Z-00-DX1.83AF88B9-C59B-48C6-A739-85ACB8F8ECA9_20x",
"TCGA-CK-5916-01Z-00-DX1.726a78b1-e64f-4dd6-8f7e-e43e98f1f453_20x",
"TCGA-CM-6674-01Z-00-DX1.4a08b16a-788e-43dc-85d2-baff6e911de2_20x",
"TCGA-A6-6651-01Z-00-DX1.09ad2d69-d71d-4fa2-9504-80557a053db4_20x",
"TCGA-NH-A50V-01Z-00-DX1.408BA0A6-E569-4464-A8CB-D6553A4DF9E0_20x",
"TCGA-A6-5659-01Z-00-DX1.c671806f-013e-4d99-9841-cda5bd43eff1_20x",
"TCGA-CK-4950-01Z-00-DX1.03dcc4c2-2b63-45a2-8561-bf18193202b5_20x",
"TCGA-CM-5860-01Z-00-DX1.95f23758-00b7-4602-b4ef-944130528f36_20x",
"TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F_20x",
"TCGA-CA-5797-01Z-00-DX1.6549a80e-4b68-4147-949b-6149ab680313_20x",
"TCGA-DM-A0XD-01Z-00-DX1.DAFA56D4-85CB-4FB1-B5BB-E993CA522FF8_20x",
"TCGA-CM-6164-01Z-00-DX1.ccf5ce96-b732-4c35-b177-d3dbe2ed89cb_20x",
"TCGA-CK-6748-01Z-00-DX1.1dd76660-7858-470c-a27b-36586b788125_20x",
"TCGA-AY-A54L-01Z-00-DX1.BD4039B4-D732-418B-9CC9-064095A1F06F_20x",
"TCGA-CM-6170-01Z-00-DX1.aa9c41ea-3894-4524-a94c-f44c6c53c2d0_20x",
"TCGA-NH-A6GC-01Z-00-DX1.29073D7E-5EEF-4BBA-96BE-DC8C69924C42_20x",
"TCGA-A6-4105-01Z-00-DX1.228b02a5-04fa-4392-bf03-b297c19665c3_20x",
"TCGA-D5-6930-01Z-00-DX1.fbf9468b-67c6-413d-a188-707ee2ab9b95_20x",
"TCGA-AD-6548-01Z-00-DX1.4e047481-8926-48e6-9eba-46597c4cc396_20x",
"TCGA-CM-6162-01Z-00-DX1.806a99a3-cda2-4dde-8d13-d22912b44d49_20x",
"TCGA-D5-6926-01Z-00-DX1.3830423a-3587-432b-9a6c-84f838e49fe6_20x",
"TCGA-AA-3715-01Z-00-DX1.24d6e746-ad61-4587-a2b9-8903331b279c_20x",
"TCGA-DM-A1HA-01Z-00-DX1.E56FC26A-DDB9-4121-9E79-5009FB23CCEB_20x"
]



modelRootPath = os.path.join(BASE_PATH,'task_1_tumor_vs_normal_s22')
modelConfig = initModelArgs(os.path.join(modelRootPath,'experiment_task_1_tumor_vs_normal.txt'),FEATURE_MODELS)
model = loadModel(modelConfig, os.path.join(modelRootPath,[i for i in os.listdir(modelRootPath) if i.endswith('.pt')][0]))

ExternalDataPath = EXTERNAL_DATA_PATH
ExternalDataPath = [os.path.join(ExternalDataPath,i) for i in os.listdir(ExternalDataPath) if i.replace('_20x.pt','_20x') in temp_names]
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
        result['svsName'].append(svsName.replace('_20x.pt',''))
        result['Predict'].append('C1' if int(Y_hat) == 0 else 'C2')

        # Y_hat就是类别名
os.makedirs(f'inference/modelPredict_{PROJECT_NAMES}',exist_ok=True)
resultPath = os.path.join(f'inference/modelPredict_{PROJECT_NAMES}',str(MULTI_INSTANCE_MODELS)+'_'+str(PROJECT_NAMES)+'_'+str(FEATURE_MODELS)+'.csv')
pd.DataFrame(result).to_csv(resultPath,index=False)
