import pyvips
import os

svsPath = '/datasets2/lizhiyong/colon/colon20x20240908'

# Angiogenesis
AngiogenesisSvsPath1 = 'TCGA-AA-3692-01Z-00-DX1.6e8c2370-54a7-4fce-b55c-bdb459828990_20x.svs' # C1
AngiogenesisSvsPath2 = 'TCGA-AA-A004-01Z-00-DX1.2576461E-7FA3-4CC6-8CC3-58D8E88CE04D_20x.svs' # C2 

# 读取图像并设置压缩选项，去除元数据
svs_image = pyvips.Image.new_from_file(os.path.join(svsPath, AngiogenesisSvsPath1), level=2)
svs_image = svs_image.copy()  # 复制图像，以便去除元数据
svs_image.write_to_file('tiff/AngiogenesisC1.tif', compression='lzw')  # 使用LZW压缩，并禁用元数据

svs_image = pyvips.Image.new_from_file(os.path.join(svsPath, AngiogenesisSvsPath2), level=2)
svs_image = svs_image.copy()  # 复制图像，去除元数据
svs_image.write_to_file('tiff/AngiogenesisC2.tif', compression='lzw')

# Autophagy
AutophagySvsPath1 = 'TCGA-AA-3524-01Z-00-DX1.b1aae264-87be-4514-8f9d-25660b39caa7_20x.svs' # C1
AutophagySvsPath2 = 'TCGA-3L-AA1B-01Z-00-DX1.8923A151-A690-40B7-9E5A-FCBEDFC2394F_20x.svs' # C2

svs_image = pyvips.Image.new_from_file(os.path.join(svsPath, AutophagySvsPath1), level=2)
svs_image = svs_image.copy()  # 复制图像，去除元数据
svs_image.write_to_file('tiff/AutophagyC1.tif', compression='lzw')

svs_image = pyvips.Image.new_from_file(os.path.join(svsPath, AutophagySvsPath2), level=2)
svs_image = svs_image.copy()  # 复制图像，去除元数据
svs_image.write_to_file('tiff/AutophagyC2.tif', compression='lzw')

# Stemness
StemnessSvsPath1 = 'TCGA-AA-3986-01Z-00-DX1.db60e495-c0eb-416c-b65b-55ce62ed10b0_20x.svs'  # C2
StemnessSvsPath2 = 'TCGA-D5-6541-01Z-00-DX1.b342c06b-8c59-4218-82f5-388568037e41_20x.svs'  # C1

svs_image = pyvips.Image.new_from_file(os.path.join(svsPath, StemnessSvsPath1), level=2)
svs_image = svs_image.copy()  # 复制图像，去除元数据
svs_image.write_to_file('tiff/StemnessC2.tif', compression='lzw')

svs_image = pyvips.Image.new_from_file(os.path.join(svsPath, StemnessSvsPath2), level=2)
svs_image = svs_image.copy()  # 复制图像，去除元数据
svs_image.write_to_file('tiff/StemnessC1.tif', compression='lzw')
