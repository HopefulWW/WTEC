import torch
import torchvision
import torch.optim
import os
import numpy as np
from PIL import Image
import glob
import time
import cv2
import matplotlib.pyplot as plt
from models import model_MSEC as model

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def exposure(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_exposure = Image.open(image_path)

	# mode, channels = get_image_mode_and_channels(data_exposure)
	# print(f'图片模式: {mode}, 通道数量: {channels}')

	# 确保图像有3个通道
	if data_exposure.mode == 'RGBA':
		# 将图像转换为RGB模式，丢弃Alpha通道
		data_exposure = data_exposure.convert('RGB')
	elif data_exposure.mode == 'CMYK':
		# 将图像转换为RGB模式，丢弃CMYK通道
		data_exposure = data_exposure.convert('RGB')
	# 或者你可以选择保留其中3个通道，例如：
	# data_lowlight = data_lowlight.split()[0]

	# mode, channels = get_image_mode_and_channels(data_exposure)
	# print(f'图片模式: {mode}, 通道数量: {channels}')

	# # 将图像调整为模型期望的尺寸
	# data_lowlight = data_lowlight.resize((270, 264))

	# 将图像转换为NumPy数组
	data_exposure = np.asarray(data_exposure)
	# 计算图像的像素均值
	# mean_value = np.mean(data_lowlight / 255.0, axis=(0, 1))

	data_exposure = data_exposure/255.0

	data_exposure = torch.from_numpy(data_exposure).float()
	data_exposure = data_exposure.permute(2,0,1)
	data_exposure = data_exposure.cuda().unsqueeze(0)
	# data_exposure = data_exposure.unsqueeze(0)
	w_net = model.WMDCNN().cuda()
	w_net = torch.nn.DataParallel(w_net)
	# w_net = model.WRB().cpu()
	w_net.load_state_dict(torch.load('snapshots-nod/512_epoch75.pth'))
	# w_net.load_state_dict(torch.load('snapshots/Epoch2.pth'))

	start = time.time()
	out_img = w_net(data_exposure,is_training=False)
	# out = w_net(data_lowlight.cpu())

	end_time = (time.time() - start)
	print(end_time)

	out_img_np = out_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
	# out_img_np = out_img.squeeze(0).permute(1, 2, 0).detach().numpy()

	# 将 NumPy 数组从 0 到 1 的范围转换为 0 到 255 的范围
	out_img_np = (out_img_np * 255).astype(np.uint8)
	out_img_rgb = cv2.cvtColor(out_img_np, cv2.COLOR_BGR2RGB)
	# 使用matplotlib显示图片需要交换颜色通道
	out_img_rgb = out_img_rgb[:, :, ::-1]

	# plt.figure(figsize=(10, 5))
	# plt.subplot(1, 2, 1)
	# plt.imshow(data_exposure.cpu().squeeze().permute(1, 2, 0))
	# plt.title("Original Image")
	# plt.subplot(1, 2, 2)
	# plt.imshow(out_img_rgb)
	# plt.title("Enhanced Image")
	# plt.show()


	# # 使用 OpenCV 创建窗口并显示图像
	# cv2.imshow("Enhanced Image", enhanced_image_rgb)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	# image_path = image_path.replace('test-720','result-noD-e75')
	image_path = image_path.replace('test', 'result')

	# Afifi test
	# image_path = image_path.replace('T', 'Lout')

	result_path = image_path
	dir, fn = os.path.split(result_path)
	if not os.path.exists(dir):
		os.makedirs(dir)

	torchvision.utils.save_image(out_img, result_path)

if __name__ == '__main__':
# test_images
	with torch.no_grad():
		# filePath = 'result/test/'
		# filePath = 'result/test/DICM/goodtest/'
		# filePath = 'G:/light/dataset/multiexposure-people/testing/test-720/'
		filePath = 'data/test/'

		#afifi test
		# filePath = 'G:/light/dataset/MultiExposure_dataset/testing/T/'
		file_list = os.listdir(filePath)

		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*")
			for image in test_list:
				# image = image
				print(image)
				exposure(image)
