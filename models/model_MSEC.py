import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse
from modelblock.unet4 import SubNet_4layers
# from modelblock.unet3 import SubNet_3layers
from modelblock.blocks import Res
import torch.nn.functional as F

class x_input(nn.Module):
    def __init__(self):
        super(x_input, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x


# 主网络模型，包含多个子网络和上采样层。该模型的前向传播过程通过组合子网络和上采样层来生成输出
class WMDCNN(nn.Module):
    def __init__(self):
        super(WMDCNN, self).__init__()
        # self.subnet5 = SubNet_5layers(3,3,24)
        # self.subnet4 = SubNet_4layers(9,9,72)
        self.subnet44 = SubNet_4layers(3,3,24)
        # self.subnet3 = SubNet_3layers(9,9,72)
        # self.subnet33 = SubNet_3layers(3,3,24)
        self.res9 = Res(9,9)
        # self.xinput = x_input()

        self.DWT = DWTForward(J=1, mode='zero', wave="db2")  # Accepts all wave types available to PyWavelets
        self.IDWT = DWTInverse(mode='zero', wave="db2")

    def forward(self, x, target=None, is_training=True):
        batch_size, _, h, w = x.shape

        # 输入 x 进行处理，确保其高度和宽度为偶数。如果高度和宽度为奇数，进行边缘填充处理
        if h % 2 == 1 and w % 2 == 1:
            pad = nn.ReplicationPad2d(1)
            x = pad(x)
            x = x[:, :, 0:-1, 0:-1]
        elif h % 2 == 1:
            pad = nn.ReplicationPad2d(1)
            x = pad(x)
            x = x[:, :, 0:-1, 1:-1]
        elif w % 2 == 1:
            pad = nn.ReplicationPad2d(1)
            x = pad(x)
            x = x[:, :, 1:-1, 0:-1]

        if is_training:
            t_yl, t_yh = self.DWT(target)
            t_yh = t_yh[0]
            t_fh, t_fw = t_yh.shape[-2], t_yh.shape[-1]
            t_yh = t_yh.view(batch_size, -1, t_fh, t_fw)


        yl, yh = self.DWT(x)  # 对输入 x 进行小波变换，得到低频分量 yl 和高频分量 yh
        yh = yh[0]  # 返回的yh是一个list
        fh, fw = yh.shape[-2], yh.shape[-1]
        yh = yh.view(batch_size, -1, fh, fw)
        # out = torch.cat((yl, yh), 1)  # 将 yl 和 yh 连接在一起形成一个输出out

        out1 = self.subnet44(yl)
        yl1 = out1+yl

        out3 = self.res9(yh)
        yh1 = out3+yh

        yl = yl1
        yh = yh1.view(batch_size, -1, 3, fh, fw)
        yh = [yh, ]

        # 将 yl 和 yh 传入逆小波变换模块 WTI 中进行逆小波变换，得到重构的输出 out
        out = self.IDWT((yl, yh))

        # out = self.xinput(out)+x

        # 根据输入 x 的高度和宽度是否为奇数，对输出 out 进行裁剪，去除填充的部分
        if h % 2 == 1 and w % 2 == 1:
            out = out[:, :, 1:, 1:]
        elif h % 2 == 1:
            out = out[:, :, 1:, :]
        elif w % 2 == 1:
            out = out[:, :, :, 1:]

        if is_training:
            return yl1, yh1, t_yl, t_yh, out
        else:
            return out

# 定义了一个判别器模型，包含多个卷积层和激活函数，用于对输入进行判别。
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(3,8,4,2,1)
        self.ac1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(8,16,4,2,1)
        self.bn2 = nn.BatchNorm2d(16)
        self.ac2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(16,32,4,2,1)
        self.ac3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(32,64,4,2,1)
        self.ac4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(64,128,4,2,1)
        self.ac5 = nn.LeakyReLU()

        self.conv6 = nn.Conv2d(128,128,4,2,1)
        self.ac6 = nn.LeakyReLU()

        self.conv7 = nn.Conv2d(128,256,4,2,1)
        self.ac7 = nn.LeakyReLU()

        self.conv8 = nn.Conv2d(256,1,2,2,0)

    def forward(self,x):
        if x.shape[2]!=256 and x.shape[3]!=256:
            x = F.interpolate(x,(256,256),mode='bilinear',align_corners=True)
        y = self.ac1(self.conv1(x))
        y = self.ac2(self.bn2(self.conv2(y)))
        y = self.ac3(self.conv3(y))
        y = self.ac4(self.conv4(y))
        y = self.ac5(self.conv5(y))
        y = self.ac6(self.conv6(y))
        y = self.ac7(self.conv7(y))
        y = self.conv8(y)
        return y