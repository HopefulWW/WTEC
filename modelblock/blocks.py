import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义了一个cnnblock类，表示一个卷积块
class cnnblock(nn.Module):
    #  创建了两个卷积层self.cnn_conv1和self.cnn_conv2，使用LeakyReLU作为激活函数，并设置inplace=True以节省内存。
    def __init__(self, in_channle, out_channle):
        super(cnnblock, self).__init__()
        self.cnn_conv1 = nn.Conv2d(in_channle, out_channle, 3, 1, 1)
        self.ac1 = nn.LeakyReLU(inplace=True)

        self.cnn_conv2 = nn.Conv2d(out_channle, out_channle, 3, 1, 1)
        self.ac2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.cnn_conv1(x)
        x = self.ac1(x)
        x = self.cnn_conv2(x)
        x = self.ac2(x)
        return x


    # 用于上采样操作
class Upsample(nn.Module):
    """Upscaling"""

    # 根据bilinear参数的取值，选择使用双线性插值（bilinear=True）或转置卷积（bilinear=False）进行上采样
    # 如果使用双线性插值，会创建一个上采样层self.up，并使用一个卷积层self.conv来减少通道数
    # 如果使用转置卷积，会创建一个转置卷积层self.up
    def __init__(self, in_channels, out_channels, bilinear=True,dropout=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.bilinear = bilinear
        self.dropout = dropout
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2)
        self.ac = nn.LeakyReLU(inplace=True)

        if self.dropout:
            self.drop_layer = nn.Dropout(p=0.1)

    def forward(self, x, shape1, shape2):
        x = self.up(x)
        # input is CHW
        # 根据输入形状和上采样结果的形状计算差异
        diffY = shape1 - x.shape[2]
        diffX = shape2 - x.shape[3]
        if self.bilinear:
            x = self.conv(x)
        x = self.ac(x)
        # 使用填充操作对上采样结果进行填充
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        if self.dropout:
            x = self.drop_layer(x)
        return x



# 注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# 残差
class ResBlock(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1,padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU()

    def forward(self,x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = residual+out
        out = self.relu(out)

        return out


class Res(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Res, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.PReLU()
        self.layer = nn.Sequential(
            ResBlock(64,64,3),
            ResBlock(64,64,3),
            ResBlock(64,64,3),
            ResBlock(64,64,3),
            ResBlock(64,64,3),
            ResBlock(64,64,3)
        )
        self.conv = nn.Conv2d(64, out_channels, kernel_size=1,stride=1)


    def forward(self,x):
        out = self.relu(self.conv3(x))
        out = self.layer(out)
        out = torch.sigmoid(self.conv(out))
        out = x+out
        return out