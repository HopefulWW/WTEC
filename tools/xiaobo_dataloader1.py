"""
定义了一个数据加载器类dataloader，用于加载训练数据集
"""

import os
import sys
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
import cv2
import os
import albumentations as A

random.seed(0)

# 定义了一个数据增强的转换函数trans，使用了albumentations库来实现增强操作
# 在列表中使用A.OneOf函数将水平翻转和垂直翻转操作组合起来
# A.OneOf函数表示在给定的一组增强操作中选择一个进行应用
# 在这里，水平翻转和垂直翻转操作被设置为等概率（p=0.5）进行选择
trans = A.Compose([
    A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], p=0.5)
    # A.GaussNoise(p=0.2),    # 将高斯噪声应用于输入图像。
    # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.3),
    # 随机应用仿射变换：平移，缩放和旋转输入
    # A.RandomBrightnessContrast(p=0.5),   # 随机明亮对比度
])


# 获取训练数据集的图像文件路径列表
# 获取低光图像和正常图像的文件路径列表
# 通过lowlight_images_path和nomal_images_path参数获取对应路径下的图像文件
# 并将它们分别存储到image_list_lowlight和image_list_nomal中
def populate_train_list(exposure_images_path, nomal_images_path):
    image_list_lowlight = glob.glob(exposure_images_path + '/*')
    # image_list_lowlight = [f for f in os.listdir(exposure_images_path) if
    #                       os.path.isfile(os.path.join(exposure_images_path, f))]
    image_list_nomal = glob.glob(nomal_images_path + '/*')
    # 将列表中的元素重复5次。这样做的目的是为了使正常图像的数量与低光图像的数量相等。这样可以保证每个低光图像都有对应的正常图像进行训练
    image_list_nomal = 5 * image_list_nomal
    image_list_lowlight.sort()
    image_list_nomal.sort()
    if len(image_list_lowlight) != len(image_list_nomal):
        print('Data length Error')
        exit()
    return image_list_lowlight, image_list_nomal


class ExposureDataset(data.Dataset):
    def __init__(self,nomal_images_path, exposure_images_path, size):
        # 获取图像文件夹路径
        self.image_list_lowlight, self.image_list_nomal = populate_train_list(
            os.path.join(exposure_images_path, 'PatchSize_' + str(size)),
            os.path.join(nomal_images_path, 'PatchSize_' + str(size)))
        # 将之前定义的参数分别赋值到类的属性，方便使用
        self.size = size
        self.trans = trans
        # 输出训练样本的总数量
        print("Total examples:", len(self.image_list_lowlight))

    # 获取指定索引index处的数据样本,并进行数据增强和分解处理
    def __getitem__(self, index):
        # 根据索引index分别获取低光图像和正常图像的文件路径
        data_lowlight_path = self.image_list_lowlight[index]
        data_nomal_path = self.image_list_nomal[index]
        # 使用cv2.imread函数加载低光图像和正常图像的数据,分别赋值给data_lowlight和data_nomal
        data_lowlight = cv2.imread(data_lowlight_path)
        data_nomal = cv2.imread(data_nomal_path)
        # 图像的像素值归一化到0到1之间
        data_lowlight = data_lowlight / 255.0
        data_nomal = data_nomal / 255.0
        # 对低光图像进行增强操作,data_lowlight作为输入图像，data_nomal作为掩码图像
        augment = self.trans(image=data_lowlight, mask=data_nomal)
        data_lowlight, data_nomal = augment['image'], augment['mask']
        # 将低频部分转换为torch.Tensor类型，并将通道维度置于第一维度
        data_lowlight = torch.from_numpy(data_lowlight).float().permute(2, 0, 1)
        data_nomal = torch.from_numpy(data_nomal).float().permute(2, 0, 1)

        return data_lowlight, data_nomal

    # 返回数据集的长度
    def __len__(self):
        return len(self.image_list_lowlight)

