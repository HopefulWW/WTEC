import torch
import torch.nn as nn
from modelblock.blocks import cnnblock, Upsample
from modelblock.blocks import CBAM



# 定义了一个包含9个卷积块和4个上采样层的子网络
class SubNet_4layers(nn.Module):
    def __init__(self,inchannl, outchannl,firstoutputchannl=64, cbam=True):
        super(SubNet_4layers, self).__init__()
        self.cbam = cbam
        self.outputchannl = outchannl
        self.block1 = cnnblock(inchannl, firstoutputchannl)
        self.maxpool = nn.MaxPool2d(2)
        self.block2 = cnnblock(firstoutputchannl, 2 * firstoutputchannl)
        self.block3 = cnnblock(2 * firstoutputchannl, 4 * firstoutputchannl)
        self.block4 = cnnblock(4 * firstoutputchannl, 8 * firstoutputchannl)
        self.block5 = cnnblock(8 * firstoutputchannl, 16 * firstoutputchannl)
        if self.cbam:
            # self.Cbam1 = CBAM(firstoutputchannl)
            # self.Cbam2 = CBAM(2 * firstoutputchannl)
            # self.Cbam3 = CBAM(4 * firstoutputchannl)
            # self.Cbam4 = CBAM(8 * firstoutputchannl)
            self.Cbam5 = CBAM(16 * firstoutputchannl)

        self.up1 = Upsample(16 * firstoutputchannl, 8 * firstoutputchannl)
        self.block6 = cnnblock(16 * firstoutputchannl, 8 * firstoutputchannl)

        self.up2 = Upsample(8 * firstoutputchannl, 4 * firstoutputchannl)
        self.block7 = cnnblock(8 * firstoutputchannl, 4 * firstoutputchannl)

        self.up3 = Upsample(4 * firstoutputchannl, 2 * firstoutputchannl)
        self.block8 = cnnblock(4 * firstoutputchannl, 2 * firstoutputchannl)

        self.up4 = Upsample(2 * firstoutputchannl, firstoutputchannl)
        self.block9 = cnnblock(2 * firstoutputchannl, firstoutputchannl)
        self.finalconv = nn.Conv2d(firstoutputchannl, self.outputchannl, 1, 1, 0)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(self.maxpool(out1))
        out3 = self.block3(self.maxpool(out2))
        out4 = self.block4(self.maxpool(out3))
        out5 = self.block5(self.maxpool(out4))
        if self.cbam:
            # out1 = self.Cbam1(out1)
            # out2 = self.Cbam2(out2)
            # out3 = self.Cbam3(out3)
            # out4 = self.Cbam4(out4)
            out5 = self.Cbam5(out5)

        in6 = torch.cat([self.up1(out5, out4.shape[2], out4.shape[3]), out4], 1)
        out6 = self.block6(in6)
        in7 = torch.cat([self.up2(out6, out3.shape[2], out3.shape[3]), out3], 1)
        out7 = self.block7(in7)
        in8 = torch.cat([self.up3(out7, out2.shape[2], out2.shape[3]), out2], 1)
        out8 = self.block8(in8)
        in9 = torch.cat([self.up4(out8, out1.shape[2], out1.shape[3]), out1], 1)
        out9 = self.block9(in9)
        predict = self.finalconv(out9)

        return predict