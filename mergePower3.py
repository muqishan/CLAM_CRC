from openslide import OpenSlide
import pyvips
import numpy as np
from math import ceil
import openslide
import os
import tifffile
import cv2
from tqdm import tqdm
import time
import glob
import copy
import time

TILE_SIZE = 512
erroR = []

os.environ['VIPS_CONCURRENCY'] = '128' 
gfi = lambda img,ind : copy.deepcopy(img[ind[0]:ind[1], ind[2]:ind[3]])

def find_file(path,depth_down,depth_up=0,suffix='.xml'):
    ret = []
    for i in range(depth_up,depth_down):
        _path = os.path.join(path,'*/'*i+'*'+suffix)
        ret.extend(glob.glob(_path))
    ret.sort()
    return ret

def up_to16_manifi(hw):
    return int(ceil(hw[0]/TILE_SIZE)*TILE_SIZE), int(ceil(hw[1]/TILE_SIZE)*TILE_SIZE)

def gen_im(wsi, index):
    # ind = 0
    # while True:
    #     temp_img = gfi(wsi, index[ind])
    #     ind+=1
    #     yield temp_img
    for ind in range(len(index)):
        temp_img = gfi(wsi, index[ind])
        yield temp_img

def get_name_from_path(file_path:str, ret_all:bool=False):
    dir, n = os.path.split(file_path)
    n, suffix = os.path.splitext(n)
    if ret_all:
        return dir, n, suffix
    return n

def gen_patches_index(ori_size, *, img_size=224, stride = 224,keep_last_size = False):

    height, width = ori_size[:2]
    index = []
    if height<img_size or width<img_size: 
        print("input size is ({} {}), small than img_size:{}".format(height, width, img_size))
        return index
        
    for h in range(0, height+1, stride):
        xe = h+img_size
        if h+img_size>height:
            xe = height
            h = xe-img_size if keep_last_size else h

        for w in range(0, width+1, stride):
            ye = w+img_size
            if w+img_size>width:
                ye = width
                w = ye-img_size if keep_last_size else w
            index.append(np.array([h, xe, w, ye]))

            if ye==width:
                break
        if xe==height:
            break
    return index

def just_ff(path:str,*,file=False,floder=True,create_floder=False, info=True):

    if file:
        return os.path.isfile(path)
    elif floder:
        if os.path.exists(path):
            return True
        else:
            if create_floder:
                try:
                    os.makedirs(path) 
                    if info:
                        print(r"Path '{}' does not exists, but created ！！".format(path))
                    return True
                except ValueError:
                    if info:
                        print(r"Path '{}' does not exists, and the creation failed ！！".format(path))
                    pass
            else:
                if info:
                    print(r"Path '{}' does not exists！！".format(path))
                return False
                

def just_dir_of_file(file_path:str, create_floder:bool=True):
    _dir = os.path.split(file_path)[0]
    return just_ff(_dir, create_floder = create_floder)


def gen_pyramid_tiff(in_file, out_file, wsi_paths_,select_level=0):

    svs_desc = 'Aperio Image Library Fake\nABC |AppMag = {mag}|Filename = {filename}|MPP = {mpp}'
    label_desc = 'Aperio Image Library Fake\nlabel {W}x{H}'
    macro_desc = 'Aperio Image Library Fake\nmacro {W}x{H}'
    # 指定mpp值
    odata = in_file
    # 获取当前图像的MPP
    tar = None
    if 'aperio.MPP' in slide.properties:
        mpp = float(slide.properties['aperio.MPP'])
        tar = 0.50 / mpp
    elif 'openslide.mpp-x' in slide.properties:
        mpp_x = float(slide.properties['openslide.mpp-x'])
        mpp_y = float(slide.properties['openslide.mpp-y'])
        tax = 0.50 / mpp_x
        tay = 0.50 / mpp_y
    else:
        erroR.append(wsi_paths_)
        print(wsi_paths_,'not mpp')
        return

    mag = 20  # 目标倍率
    mpp = 0.5 
    # 换算mpp值到分辨率
    resolution = [10000 / mpp, 10000 / mpp, 'CENTIMETER']


    if odata.properties.get('aperio.Filename') is not None:
        filename = odata.properties['aperio.Filename']
    else:
        # filename = get_name_from_path(in_file)
        filename = os.path.splitext(os.path.basename(wsi_path))[0]


    image_py = pyvips.Image.new_from_file(in_file._filename, level=select_level)
    

    image = np.array(image_py)[..., 0:3]
    # print(f'origin size:{image.shape[0:2]}')
    if tar is not None:
        image = cv2.resize(image, (int(image.shape[1] // tar), int(image.shape[0] // tar)))
    else:
        image = cv2.resize(image, (int(image.shape[1] // tax), int(image.shape[0] // tay)))
    
    # print(f"finish loading '{in_file}'. costing time:{time.time()-start}")

    # 缩略图
    thumbnail_im = np.zeros([762, 762, 3], dtype=np.uint8)
    thumbnail_im = cv2.putText(thumbnail_im, 'thumbnail', (thumbnail_im.shape[1]//4, thumbnail_im.shape[0]//2), cv2.FONT_HERSHEY_PLAIN, 6, color=(255, 0, 0), thickness=3)
    # 标签图
    label_im = np.zeros([762, 762, 3], dtype=np.uint8)
    label_im = cv2.putText(label_im, 'label', (label_im.shape[1]//4, label_im.shape[0]//2), cv2.FONT_HERSHEY_PLAIN, 6, color=(0, 255, 0), thickness=3)
    # 宏观图
    macro_im = np.zeros([762, 762, 3], dtype=np.uint8)
    macro_im = cv2.putText(macro_im, 'macro', (macro_im.shape[1]//4, macro_im.shape[0]//2), cv2.FONT_HERSHEY_PLAIN, 6, color=(0, 0, 255), thickness=3)

    # tile 大小
    tile_hw = np.int64([TILE_SIZE, TILE_SIZE])

    width, height = image.shape[0:2]
    # print(f'new size:{width,height}')
    # 要需要的金字塔分辨率
    multi_hw = np.int64([(width, height), # 标准的mpp0.5 分辨率20x
                          (width//16, height//16), 
                         (width//32, height//32)])

    # 尝试写入 svs 格式
    with tifffile.TiffWriter(out_file, bigtiff=True) as tif:
        thw = tile_hw.tolist()
        # outcolorspace 要保持为默认的 YCbCr，不能使用rgb，否则颜色会异常
        # 95 是默认JPEG质量，值域是 0-100，值越大越接近无损
        compression = ['JPEG', 95, dict(outcolorspace='YCbCr')]
        # compression = 'JPEG'
        kwargs = dict(subifds=0, photometric='rgb', planarconfig='CONTIG', compression=compression, dtype=np.uint8, metadata=None)

        for i, hw in enumerate(multi_hw):
            hw = up_to16_manifi(hw)
            temp_wsi = cv2.resize(image, (hw[1], hw[0]))
            new_x, new_y = up_to16_manifi(hw)
            new_wsi = np.ones((new_x, new_y, 3), dtype=np.uint8) * 255
            new_wsi[0:hw[0], 0:hw[1], :] = temp_wsi[..., 0:3]
            index = gen_patches_index((new_x, new_y), img_size=TILE_SIZE, stride=TILE_SIZE)
            gen = gen_im(new_wsi, index)

            if i == 0:
                desc = svs_desc.format(mag=mag, filename=filename, mpp=mpp)
                # tif.write(data=gen, resolution=resolution, description=desc, **kwargs)
                tif.write(data=gen, shape=(*hw, 3), tile=thw[::-1], resolution=resolution, description=desc, **kwargs)
                _hw = up_to16_manifi(multi_hw[-2])
                thumbnail_im = cv2.resize(image, (_hw[1], _hw[0]))[..., 0:3]
                tif.write(data=thumbnail_im, description='', **kwargs)
            else:
                tif.write(data=gen, shape=(*hw, 3), tile=thw[::-1], resolution=resolution, description='', **kwargs)
        _hw = up_to16_manifi(multi_hw[-2])
        macro_im = cv2.resize(image, (_hw[1], _hw[0]))[..., 0:3]
        tif.write(data=macro_im, subfiletype=9, description=macro_desc.format(W=macro_im.shape[1], H=macro_im.shape[0]), **kwargs)



if __name__ == "__main__":

    origin_path = ''       # origin TCGA Slide Path
    tar_path = ''     # save path
    os.makedirs(tar_path, exist_ok=True) 

    wsi_paths = [os.path.join(origin_path, i) for i in os.listdir(origin_path) if i.endswith('.svs') or i.endswith('mrxs')]

    for wsi_path in tqdm(wsi_paths):
        time.sleep(0.1)
        output_path = os.path.join(tar_path, os.path.splitext(os.path.basename(wsi_path))[0] + '_20x.svs')
        if os.path.exists(output_path):
            continue
        slide = openslide.open_slide(wsi_path)
        gen_pyramid_tiff(slide, output_path,wsi_paths)


