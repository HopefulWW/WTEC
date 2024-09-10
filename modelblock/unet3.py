import torch
import torch.nn as nn

from modelblock.blocks import cnnblock, Upsample
# 定义了一个包含7个卷积块和3个上采样层的子网络
class SubNet_3layers(nn.Module):
    def __init__(self, inchannl, outchannl,firstoutputchannl=64):
        super(SubNet_3layers, self).__init__()
        # 输出通道为3
        self.outputchannl = outchannl
        # 最大池化、7个卷积块、3个上采样层、最后的卷积层
        self.maxpool = nn.MaxPool2d(2)
        self.block1 = cnnblock(inchannl, firstoutputchannl)
        self.block2 = cnnblock(firstoutputchannl, 2 * firstoutputchannl)
        self.block3 = cnnblock(2 * firstoutputchannl, 4 * firstoutputchannl)
        self.block4 = cnnblock(4 * firstoutputchannl, 8 * firstoutputchannl)
        self.up1 = Upsample(8 * firstoutputchannl, 4 * firstoutputchannl)
        self.block5 = cnnblock(8 * firstoutputchannl, 4 * firstoutputchannl)
        self.up2 = Upsample(4 * firstoutputchannl, 2 * firstoutputchannl)
        self.block6 = cnnblock(4 * firstoutputchannl, 2 * firstoutputchannl)
        self.up3 = Upsample(2 * firstoutputchannl, firstoutputchannl)
        self.block7 = cnnblock(2 * firstoutputchannl, firstoutputchannl)
        self.finalconv = nn.Conv2d(firstoutputchannl, self.outputchannl, 1, 1, 0)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(self.maxpool(out1))
        out3 = self.block3(self.maxpool(out2))
        out4 = self.block4(self.maxpool(out3))
        in5 = torch.cat([self.up1(out4, out3.shape[2], out3.shape[3]), out3], 1)
        out5 = self.block5(in5)
        in6 = torch.cat([self.up2(out5, out2.shape[2], out2.shape[3]), out2], 1)
        out6 = self.block6(in6)
        in7 = torch.cat([self.up3(out6, out1.shape[2], out1.shape[3]), out1], 1)
        out7 = self.block7(in7)
        predict = self.finalconv(out7)
        return predict