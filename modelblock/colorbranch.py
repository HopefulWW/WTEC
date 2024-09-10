import torch
import torch.nn as nn

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



# 定义颜色分支
class ColorBranch(nn.Module):
    def __init__(self):
        super(ColorBranch, self).__init__()
        # 通过卷积层逐渐提取特征，最后一个卷积层输出3个通道的颜色特征
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.features(x)