import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
import pywt
from pytorch_msssim import SSIM
from pytorch_wavelets import DWTForward, DWTInverse


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vgg = vgg16(pretrained=True).features[:9].to(device)  # 选取VGG的前9层
        self.vgg = vgg.eval()
        self.L1_loss = L1Loss()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, out, target):
        # 提取两个图像在VGG的特征
        feature_gen = self.vgg(out)
        feature_target = self.vgg(target)

        # 计算这些特征的MSE损失
        l1_loss = self.L1_loss(feature_gen, feature_target)

        return l1_loss


# L1损失 构造平滑L1损失公式：sqrt[(x-y)² + ε],最后返回损失的总和
class L1Loss(nn.Module):

    def __init__(self, eps=1e-6):
        super(L1Loss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        l1_loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        l1_loss =  l1_loss / (torch.numel(x) + torch.numel(y))
        return l1_loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()  # 调用 nn.Module 的构造函数
        self.mse_loss = nn.MSELoss()

    def forward(self, out, target):
        # 计算并返回两张图像之间的 MSE 损失
        mse_loss = self.mse_loss(out, target)
        return mse_loss


# 定义小波系数损失
class WaveletLoss(nn.Module):
    def __init__(self, wavelet='haar', level=1):
        super(WaveletLoss, self).__init__()
        self.wavelet = wavelet
        self.level = level
        self.DWT = DWTForward(J=1, mode='zero', wave="Haar").cuda()   # Accepts all wave types available to PyWavelets

    def forward(self, out_yl,out_yh, t_yl,t_yh,):

        # 初始化损失值
        wave_loss = 0.0
        L1 = L1Loss()
        mse = MSELoss()

        l_diff = L1(out_yl, t_yl)
        wave_loss += l_diff

        h_diff = L1(out_yh, t_yh)
        wave_loss += h_diff

        # wave_loss = wave_loss / (torch.numel(out_yl) + torch.numel(out_yh))
        return wave_loss


# 计算对抗损失
class Advloss(nn.Module):
    def __init__(self ,weight=1.0):
        super(Advloss,self).__init__()
        self.weight = weight
    def forward(self,P_out):
        adv_loss = -self.weight*torch.mean(torch.log(torch.sigmoid(P_out)+1e-9))
        return adv_loss


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()  # 调用 nn.Module 的构造函数
        self.ssim_loss = SSIM()

    def forward(self, out, target):
        # 计算并返回两张图像之间的ssim 损失
        ssim_loss = 1 - self.ssim_loss(out, target)
        # print("mse_loss =", mse_loss)
        return ssim_loss


class Colorloss(nn.Module):
    def __init__(self):
        super(Colorloss, self).__init__()
        self.cos = torch.nn.CosineSimilarity(1, 1e-8)

    def forward(self, output, target):
        col_loss = (1 - self.cos(output, target).mean()) * 0.5

        return col_loss



# 光照平滑损失
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, output):
     # 获取x的维度信息
        batch_size = output.size()[0]
        h_x = output.size()[2]
        w_x = output.size()[3]
        # 确定水平方向和垂直方向上需要计算差异总数
        count_h = (output.size()[2] - 1) * output.size()[3]
        count_w = output.size()[2] * (output.size()[3] - 1)
         # x 是一个形状为 (batch_size, channels, height, width)
        h_tv = torch.pow((output[:, :, 1:, :] - output[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((output[:, :, :, 1:] - output[:, :, :, :w_x - 1]), 2).sum()

        tv_loss = self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

        return tv_loss

# 计算总损失
class Myloss(nn.Module):
    def __init__(self):
        super(Myloss,self).__init__()
        self.wave_Loss = WaveletLoss()
        self.ssim_loss = SSIMLoss()
        # self.mse_loss = MSELoss()
        self.adv_loss = Advloss()
        self.col_loss = Colorloss()
        # self.L1_loss = L1Loss()
        self.VGGLoss = VGGLoss()
        # self.Tvloss = TVLoss()

    # def forward(self, out, target, out_yl, out_yh, t_yl, t_yh):
    def forward(self,out,target,out_yl,out_yh, t_yl,t_yh, P_out=None, withoutadvloss = True):
        waveloss = 10 * self.wave_Loss(out_yl,out_yh, t_yl,t_yh)
        # recloss = self.VGGLoss(out, target) + self.ssim_loss(out , target)
        recloss = 15 * (self.VGGLoss(out, target)+ self.ssim_loss(out , target))
        # tvloss = 20 * self.Tvloss(out)
        # advloss = self.adv_loss(P_out)
        colloss = 60 * self.col_loss(out,target)
        if withoutadvloss:
            myloss = waveloss + recloss+ colloss
            return recloss, waveloss,colloss,myloss

        else:
            advloss =self.adv_loss(P_out)
            myloss = waveloss + recloss + advloss + colloss
            return recloss, waveloss,colloss, advloss, myloss


class Dloss(nn.Module):
    def __init__(self):
        super(Dloss,self).__init__()
    def forward(self,P_out, P_T):
        d_loss = -torch.mean(torch.log(torch.sigmoid(P_T) + 1e-9)) - torch.mean(torch.log(1 - torch.sigmoid(P_out) + 1e-9))
        return d_loss





# # 定义小波系数损失
# class WaveletLoss(nn.Module):
#     def __init__(self, wavelet='haar', level=1):
#         super(WaveletLoss, self).__init__()
#         self.wavelet = wavelet
#         self.level = level
#         # J为分解的层次数,wave表示使用的变换方法
#         self.DWT = DWTForward(J=1, mode='zero', wave="Haar").cuda()   # Accepts all wave types available to PyWavelets
#
#     def forward(self, out_yl,out_yh, t_yl,t_yh,):
#
#         # 初始化损失值
#         wave_loss = 0.0
#         #
#         # o_yl, o_yh = self.DWT(out)  # 对输入 x 进行小波变换，得到低频分量 yl 和高频分量 yh
#         # t_yl, t_yh = self.DWT(target)  # 对输入 x 进行小波变换，得到低频分量 yl 和高频分量 yh
#         #
#         # l_diff = torch.sum((out_yl - t_yl) ** 2)
#         # wave_loss += l_diff
#         criterion = L1Loss()
#         l_diff = criterion(out_yl, t_yl)
#         wave_loss += l_diff
#
#         # h_diff = torch.sum((out_yh - t_yh) ** 2)
#         h_diff = criterion(out_yh, t_yh)
#         wave_loss += h_diff
#
#         wave_loss = wave_loss / (torch.numel(out_yl) + torch.numel(out_yh))
#
#         # # 由于小波变换是在CPU上操作，所以必须将张量移至CPU上，并转为numpy格式
#         # rec_np = out.squeeze().cpu().detach().numpy()
#         # tgt_np = target.squeeze().cpu().detach().numpy()
#         #
#         # # 初始化损失值
#         # wave_loss = 0.0
#         #
#         # # 对每个样本分别进行小波变换
#         # l_rec, h_rec = pywt.wavedec2(rec_np, self.wavelet, level=self.level)
#         # l_tgt, h_tgt = pywt.wavedec2(tgt_np, self.wavelet, level=self.level)
#         #
#         # # 归一化低频系数并计算其平方和
#         # l_rec = torch.tensor(l_rec, dtype=torch.float32) / 255.0
#         # l_tgt = torch.tensor(l_tgt, dtype=torch.float32) / 255.0
#         # l_diff = torch.sum((l_rec - l_tgt)**2)
#         # wave_loss += l_diff
#         # total_elements = l_rec.numel()  # 初始化元素计数
#         #
#         # # 对高频系数也进行归一化，并计算其平方和
#         # for h_rec_level, h_tgt_level in zip(h_rec, h_tgt):
#         #     # h_rec_level 和 h_tgt_level 是包含三个numpy数组的元组(HH, HL, LH)
#         #     for rec_detail, tgt_detail in zip(h_rec_level, h_tgt_level):
#         #         # 归一化高频系数并计算其平方和
#         #         rec_detail_tensor = torch.tensor(rec_detail, dtype=torch.float32) / 255.0
#         #         tgt_detail_tensor = torch.tensor(tgt_detail, dtype=torch.float32) / 255.0
#         #         # 计算这两个张量之间差
#         #         diff = torch.sum((rec_detail_tensor - tgt_detail_tensor)**2)
#         #         # 累加到 wave_loss
#         #         wave_loss += diff
#         #         # 记录参与计算的元素数
#         #         # total_elements += rec_detail_tensor.numel()
#         # wave_loss = (wave_loss / total_elements)
#         return wave_loss




# # 定义你的总损失函数
# class CustomLoss(nn.Module):
#     def __init__(self, wavelet='db1', level=1, ):
#         super(CustomLoss, self).__init__()
#         self.mse_loss = nn.MSELoss()
#         self.wavelet_loss = WaveletLoss(wavelet, level)
#         self.col_loss = Colorloss()
#
#     def forward(self, output, target):
#         # 计算两张图片直接的损失值
#         pixel_loss = self.mse_loss(output, target)
#         # 计算小波系数的损失值
#         wavelet_loss = self.wavelet_loss(output, target)
#         col_loss = self.col_loss(output, target)
#         # 将两个损失结合起来
#         total_loss = pixel_loss + wavelet_loss + col_loss
#
#         return total_loss



# # 颜色恒定损失
# class L_color(nn.Module):
#
#     def __init__(self):
#         super(L_color, self).__init__()
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         # 计算输入张量 x 在图像高度和宽度维度上的平均值，即对每个通道的图像像素在空间维度上求平均，输出维度（b，c，1,1）
#         mean_rgb = torch.mean(x, [2, 3], keepdim=True)
#         # 将 mean_rgb 张量沿着第1维（通道维度）进行切分，得到三个张量 mr、mg 和 mb，每个张量都具有形状 (b, 1, 1, 1)，分别表示红色通道、绿色通道和蓝色通道的平均值
#         mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
#         # 计算平均值之间的差异的平方，分别得到 Drg、Drb 和 Dgb 张量，表示红绿通道、红蓝通道和绿蓝通道之间的差异
#         Drg = torch.pow(mr - mg, 2)
#         Drb = torch.pow(mr - mb, 2)
#         Dgb = torch.pow(mb - mg, 2)
#
#         k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
#         return k




# # 空间一致性损失
# class L_spa(nn.Module):
#
#     def __init__(self):
#         super(L_spa, self).__init__()
#         # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
#
#         # 定义了四个卷积核，将这些卷积核转换为 nn.Parameter 对象，并设置为不可训练的参数
#         kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
#         kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
#         kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
#         kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
#         # requires_grad=False 指定了参数对象的 requires_grad 属性为 False，表示该参数在反向传播过程中不需要进行梯度计算，即不参与模型的训练过程
#         self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
#         self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
#         self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
#         self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
#         # 创建了一个 AvgPool2d 池化层，用于对图像均值进行池化操作
#         self.pool = nn.AvgPool2d(4)
#
#     # forward 方法是该自定义模块的前向传播函数，接受两个输入参数 org 和 enhance，分别表示原始图像和增强图像
#     def forward(self, org, enhance):
#         b, c, h, w = org.shape
#         # 计算原始图像和增强图像的通道维度上的均值
#         org_mean = torch.mean(org, 1, keepdim=True)
#         enhance_mean = torch.mean(enhance, 1, keepdim=True)
#         # 对均值图像进行池化操作
#         org_pool = self.pool(org_mean)
#         enhance_pool = self.pool(enhance_mean)
#         # 计算权重差值，使用 torch.max 函数计算权重差值，确保其大于等于指定阈值
#         weight_diff = torch.max(
#             torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
#                                                               torch.FloatTensor([0]).cuda()),
#             torch.FloatTensor([0.5]).cuda())
#         # 计算 E_1，即增强图像与原始图像均值图像的差异，使用 torch.mul 函数计算符号函数的乘积
#         E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)
#
#         # 使用卷积操作 F.conv2d 分别对原始图像和增强图像的池化结果进行卷积操作，计算 D_org_* 和 D_enhance_*
#         D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
#         D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
#         D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
#         D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)
#
#         D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
#         D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
#         D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
#         D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)
#
#         # 原始图像和增强图像在不同方向上的卷积结果之差的平方
#         D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
#         D_right = torch.pow(D_org_right - D_enhance_right, 2)
#         D_up = torch.pow(D_org_up - D_enhance_up, 2)
#         D_down = torch.pow(D_org_down - D_enhance_down, 2)
#         E = (D_left + D_right + D_up + D_down)
#         # E = 25*(D_left + D_right + D_up +D_down)
#         return E

#
# # 曝光损失
# class L_exp(nn.Module):
#     # patch_size 是平均池化操作的窗口大小，mean_val 是用于计算差异的平均值
#     def __init__(self, patch_size, mean_val):
#         super(L_exp, self).__init__()
#         # print(1)
#         # 创建了一个平均池化层 self.pool，使用 nn.AvgPool2d 平均池化操作，并将窗口大小设置为 patch_size
#         self.pool = nn.AvgPool2d(patch_size)
#         self.mean_val = mean_val
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         # 对输入特征图 x 在通道维度上求均值，得到形状为 (batch_size, 1, h, w) 的特征图。这是为了将输入特征图转换为单个通道
#         x = torch.mean(x, 1, keepdim=True)
#         # 通过 self.pool(x) 对特征图进行平均池化操作，得到形状为 (batch_size, 1, h', w') 的池化后的特征图
#         mean = self.pool(x)
#         # 计算池化后的特征图与 self.mean_val 之间的差异。具体而言，它计算了差异的平方的均值
#         d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
#         return d







