import os
import h5py
import pandas as pd

'''
patches中的坐标为补丁级切片的坐标，其中混杂了癌和非癌
Predicts为癌和非癌的二分类信息，利用Predicts的信息过滤patches的信息
'''
h5_paths = [os.path.join('ExternalDataCrc/My_RESULTS_DIRECTORYdata2/patches_origin',i) for i in os.listdir('ExternalDataCrc/My_RESULTS_DIRECTORYdata2/patches_origin')]
predicts_paths = 'ExternalDataCrc/My_RESULTS_DIRECTORYdata2/predicts'


for h5_path in h5_paths:
    with h5py.File(h5_path, 'r') as file:
        # 读取补丁坐标
        # patche_list = file['coords'][:]
        # 读取属性
        patch_level = file['coords'].attrs['patch_level']
        patch_size = file['coords'].attrs['patch_size']

    predict_path = os.path.join(predicts_paths,h5_path.split('/')[-1][:23]+'.csv')
    predict_result = pd.read_csv(predict_path)

    cancer_coords = predict_result[predict_result['predictions'] == 1][['pos_x', 'pos_y']].values


    with h5py.File(h5_path.replace('patches_origin','patches'), 'w') as new_file:
    # 创建新数据集并写入癌症坐标
        cancer_coords_dataset = new_file.create_dataset('coords', data=cancer_coords, compression="gzip", compression_opts=9)
        # 写入属性
        cancer_coords_dataset.attrs['patch_level'] = patch_level
        cancer_coords_dataset.attrs['patch_size'] = patch_size
    # print()
    # break