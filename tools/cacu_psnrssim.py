import os
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
import numpy as np
from glob import glob
import cv2
import pandas as pd
#
def calculate_psnr_ssim(img_dir,gtimg_dir):
    imgpaths = glob(img_dir+'/*')
    # gtpaths = glob(gtimg_dir+'/*')*2
    gtpaths = glob(gtimg_dir + '/*') * 5
    imgpaths.sort()
    gtpaths.sort()
    psnr_list = []
    ssim_list = []
    print('Image num:',len(imgpaths))
    for n in range(len(imgpaths)):
        img = cv2.imread(imgpaths[n])
        img_name = os.path.basename(imgpaths[n])
        gtimg = cv2.imread(gtpaths[n])
        # print(img.shape)
        # print(gtimg.shape)
        assert img.shape == gtimg.shape
        psnr = calculate_psnr(img,gtimg)
        ssim = calculate_ssim( img,gtimg,channel_axis=-1)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        image_ = img_name
        psnr_ = psnr
        ssim_ = ssim
        list_ = [image_, psnr_, ssim_]
        data = pd.DataFrame([list_])
        # data.to_csv("psnrssim720-att543.csv", mode="a", header=False, index=False)
        data.to_csv("psnrssim1.csv", mode="a", header=False, index=False)
    return psnr_list,ssim_list


if __name__ == '__main__':

    # psnrssim_file_path = "psnrssim720-att543.csv"
    psnrssim_file_path = "psnrssim1.csv"
    if not os.path.isfile(psnrssim_file_path):
        # 如果文件不存在，创建一个新的DataFrame并为其添加表头
        df_headers = pd.DataFrame(columns=["image", "PSNR", "SSIM"])
        df_headers.to_csv(psnrssim_file_path, mode='w', header=True, index=False)

    # gtimg_dir = "../data/gt"
    # img_dir = '../data/result/DICM'
    #
    # gtimg_dir = '../result/gt1'
    # img_dir = '../result/result/DICM'

    # gtimg_dir = '../result/gt1/good'
    # img_dir = '../result/result/DICM/goodresult/DICM'

    # img_dir = "G:/dataset/light/MultiExposure_dataset/testing/output"

    gtimg_dir = "G:/light/dataset/multiexposure-people/testing/720-gt"
    img_dir = 'G:/light/dataset/multiexposure-people/testing/result-noD-e75/DICM'

    # afifi test
    # gtimg_dir = "G:/light/dataset/MultiExposure_dataset/testing/expert_c_testing_set"
    # img_dir = 'G:/light/dataset/MultiExposure_dataset/testing/xiaobo-out/test'

    psnr,ssim = calculate_psnr_ssim(img_dir,gtimg_dir)
    print('PSNR:', np.mean(psnr), 'SSIM:', np.mean(ssim))
    print("psnr:",psnr)
    print("ssim:",ssim)
