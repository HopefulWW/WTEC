import torch
import torch.nn as nn
import torch.optim
import os
import argparse
from torch.utils.data import DataLoader
from tools import xiaobo_dataloader1 as dataloader
from models import model_MSEC as model
import xiaobo_loss01 as xloss
import matplotlib.pyplot as plt
import torchvision
import random
import numpy as np
import torch.backends.cudnn as cudnn
import pandas as pd

torch.cuda.empty_cache()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 设置随机数种子
# 可以在相同的种子下得到相同的随机结果，从而使实验具有可重复性，方便调试和结果比较
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# 保存最好模型
class ModelSaver:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.best_models = []

    def save_model(self, model, loss, epoch):
        model_path = os.path.join(self.model_dir, f"Epoch_{epoch + 1}.pth")
        if len(self.best_models) < 2:
            torch.save(model.state_dict(), model_path)
            self.best_models.append((model_path, loss))
        else:
            # 如果best_models已满，则找出损失最高的模型进行比较。
            worst_model_path, worst_loss = max(self.best_models, key=lambda x: x[1])
            if loss < worst_loss:
                # 新模型损失小于已保存的最差模型损失，替换之。
                os.remove(worst_model_path)
                torch.save(model.state_dict(), model_path)
                # 更新最佳模型列表，替换掉最差模型的记录。
                self.best_models.remove((worst_model_path, worst_loss))
                self.best_models.append((model_path, loss))


# 初始化神经网络的权重和偏置
# 模块 m 作为参数，如果 m 是 nn.Conv2d 或 nn.Linear 类型的实例，即卷积层或线性层
# 就会对其权重进行 Kaiming 正态分布初始化，并将偏置初始化为常数 0
def weights_init(m):
    classname = m.__class__.__name__

    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

        # 如果当前层是批归一化层
    elif classname.find('BatchNorm') != -1:
        # 使用正态分布初始化批归一化层的权重
        m.weight.data.normal_(1.0, 0.02)
        # 设置偏置项为0
        m.bias.data.fill_(0)
        # 在神经网络的前向传播中，对于输入x、权重w和偏置项b，神经元的输出可以表示为：输出 = 激活函数(w⋅x + b)
#
# def weights_init(m):
#     # 这段代码是一个权重初始化函数，用于初始化神经网络模型中的卷积层和批归一化层的权重和参数
#
#     # 获取当前层的类名
#     classname = m.__class__.__name__
#
#     # 如果当前层是卷积层
#     if classname.find('Conv') != -1:
#         # 使用正态分布（均值为0，标准差为0.02）初始化卷积层的权重
#         m.weight.data.normal_(0.0, 0.02)
#
#     # 如果当前层是批归一化层
#     elif classname.find('BatchNorm') != -1:
#         # 使用正态分布初始化批归一化层的权重
#         m.weight.data.normal_(1.0, 0.02)
#         # 设置偏置项为0
#         m.bias.data.fill_(0)
#         # 在神经网络的前向传播中，对于输入x、权重w和偏置项b，神经元的输出可以表示为：输出 = 激活函数(w⋅x + b)

def train(config):
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    # 初始化模型并应用权重初始化
    # 创建了一个未经池化的增强网络模型,这个模型通常用于图像增强任务,并加载到gpu中。
    w_net = model.WMDCNN().cuda()
    w_net = torch.nn.DataParallel(w_net)  # 多显卡

    # 如果配置文件中的use_advloss为True，则创建Discriminator模型的实例
    if config.use_advloss:
        Dnet = model.Discriminator().cuda()
        Dnet = torch.nn.DataParallel(Dnet)

    # 如果需要，加载预训练模型权重
    # 检查是否需要加载预训练的模型权重
    if config.load_pretrain == True:
        # 从配置中获取预训练模型的文件路径
        pretrain_dir = config.pretrain_dir
        # 使用 PyTorch 的 load_state_dict 函数加载预训练权重
        w_net.load_state_dict(torch.load(pretrain_dir))
        # 这段代码的目的是检查配置文件中是否需要加载预训练的模型权重，并在需要时将它们加载到神经网络模型DCE_net中。这可以在迁移学习或继续训练模型时非常有用。
        if config.use_advloss:
            Dnet.load_state_dict(torch.load(config.D_pretrain_dir))
    # 这段代码的目的是检查配置文件中是否需要加载预训练的模型权重，并在需要时将它们加载到神经网络模型DCE_net中。这可以在迁移学习或继续训练模型时非常有用。

    else:
        config.start = [0, 0]
        # 这行代码应用了权重初始化函数 weights_init到模型的所有层
        w_net.apply(weights_init)
        if config.use_advloss:
            Dnet.apply(weights_init)

    # 初始化优化器
    # 定义了一个 Adam 优化器，并将 DCE_net 模型的参数传递给优化器进行优化
    # DCE_net.parameters()：表示需要优化的参数；lr=config.lr：学习率参数，指定优化器的学习率；weight_decay=config.weight_decay：权重衰减参数，指定优化器的权重衰减系数
    optimizer = torch.optim.Adam(w_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    # 设置模型为训练模式
    w_net.train()

    # df = pd.DataFrame(columns=["range", "epoch", "epochloss"])
    # df.to_csv("data/loss/loss.csv",index=False)
    if config.use_advloss:
        D_optimizer = torch.optim.Adam(Dnet.parameters(), lr=config.D_lr, weight_decay=config.weight_decay,
                                       betas=(0.9, 0.999))
        # d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(D_optimizer, 'min', factor=0.5, patience=5)

        Dnet.train()
        d_loss =xloss.Dloss()

    # 不同的size
    for i in range(2, 3):
        # 打印当前阶段的索引
        print("----------------------i:", i, "--size:", config.sizelist[i], "----------------------")
        # 创建训练数据集
        train_dataset = dataloader.ExposureDataset(config.normal_images_path,
                                                   config.exposure_images_path,
                                                # config.exposure_suffixes,
                                               # transform=dataloader.transform
                                               size = config.sizelist[i])
        # dataloader.lowlight_loader函数创建train_dataset，该函数返回一个数据集对象，该对象包含了从指定路径config.lowlight_images_path中加载的低光照图像数据。


        # 创建数据加载器，使训练数据能够以批次的方式被加载到模型中，从而进行训练
        # 创建了一个train_loader对象，用于在训练过程中加载训练数据
        # shuffle: 这个参数指定是否在每个epoch开始时对训练数据进行洗牌（随机排序）。洗牌可以增加训练的随机性，有助于模型更好地学习数据的统计特征
        train_loader = DataLoader(train_dataset,
                                  batch_size=config.train_batch_size_list[i],
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True)

        # # 定义损失函数
        # waveloss = xloss.WaveletLoss()

        myloss= xloss.Myloss()
        loss_curve = []
        # saver = ModelSaver(config.best_snapshots_folder)

        for epoch in range(53,config.num_epochs_list[i]):
            epoch_loss=0.0
            min_loss = float('inf')
            #  iteration的计算是 样本数/批次

            print("----------------------epoch ", epoch+1," / ", config.num_epochs_list[i] , "----------------------")

            if i == 0 and epoch + 1 == 10:
                config.lr = 0.5 * config.lr
                optimizer = torch.optim.Adam(w_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            if ((i == 1) or (i == 2)) and epoch==0:
                config.lr = 0.0001
                optimizer = torch.optim.Adam(w_net.parameters(), lr=config.lr,
                                             weight_decay=config.weight_decay)

            if((i == 1) or (i == 2))and(epoch+1)% 10 == 0:
                 config.lr = 0.5 * config.lr
                 optimizer = torch.optim.Adam(w_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

                 if config.use_advloss:
                     config.D_lr = 0.5 * config.D_lr
                     D_optimizer = torch.optim.Adam(Dnet.parameters(), lr=config.D_lr, weight_decay=config.weight_decay)

            print("lr=", config.lr)
            # 如果当前阶段小于start[0]或者等于start[0]且迭代次数小于start[1]，则跳过当前迭代
            if i < config.start[0] or (i == config.start[0] and epoch < config.start[1]):
                continue

            for iteration, (img_exposure , img_normal)  in enumerate(train_loader):
                img_exposure = img_exposure.cuda()
                target = img_normal.cuda()

                # 如果使用对抗损失（use_advloss为True），则训练鉴别器
                if config.use_advloss:
                    # train D
                    D_optimizer.zero_grad()  # 将鉴别器优化器的梯度清零
                    # 计算模型生成的低光图像和目标图像的鉴别器输出，并计算鉴别器损失
                    out_yl, out_yh, t_yl, t_yh, out = w_net(img_exposure, target, is_training=True)
                    P_out = Dnet(out)
                    P_T = Dnet(target)
                    dloss = d_loss(P_out, P_T)
                    # 进行反向传播和优化
                    dloss.backward()
                    D_optimizer.step()

                # 梯度清零，进行反向传播，梯度裁剪，更新模型权重
                optimizer.zero_grad()

                # 前向传播，生成增强图像
                # model_msec
                out_yl,out_yh,t_yl,t_yh,out = w_net(img_exposure, target, is_training=True)

                if config.use_advloss:
                    P_out = Dnet(out)
                    loss_rec, loss_wave,loss_col, advloss, loss = myloss(out, target, out_yl, out_yh, t_yl, t_yh,P_out, withoutadvloss = False)
                    # loss_rec, loss_col, advloss, loss = myloss(out, target, out_yl, out_yh, t_yl,
                    #                                                       t_yh, P_out, withoutadvloss=False)

                    loss_group = {
                                        "loss_wave": loss_wave.item(),
                                        "loss_rec": loss_rec.item(),
                                        "loss_col": loss_col.item(),
                                        "adv_loss": advloss.item()
                                    }
                else:
                    loss_rec, loss_wave, loss_col, loss = myloss(out, target, out_yl, out_yh, t_yl, t_yh,withoutadvloss=True)
                    loss_group = {
                                        "loss_wave": loss_wave.item(),
                                        "loss_col": loss_col.item(),
                                        "loss_rec": loss_rec.item(),
                                 }

                loss.requires_grad_(True)
                loss.backward()
                # torch.nn.utils.clip_grad_norm(w_net.parameters(), config.grad_clip_norm)
                optimizer.step()

                if ((iteration + 1) % config.display_iter) == 0:
                    print("G_Loss at iteration", iteration + 1, ":", loss.item(),
                        # " ", "D_loss", dloss.item(),
                        " ", loss_group)

                epoch_loss+=loss.item()
                # loss_curve.append(loss.item())

                # scheduler.step(loss)
                # d_scheduler.step(loss)

                # 保存模型权重
                if ((iteration + 1) % config.snapshot_iter) == 0:
                    if loss.item() < min_loss:
                        min_loss = loss.item()
                        torch.save(w_net.state_dict(), config.snapshots_folder +str(config.sizelist[i])+ "_epoch" + str(epoch+1) + '.pth')
                        if config.use_advloss:
                            torch.save(Dnet.state_dict(), config.D_snapshots_folder+str(config.sizelist[i]) + "Dnet_epoch" + str(epoch + 1) + '.pth')

                torchvision.utils.save_image(out[:,[2,1,0],:,:], './run-out/' + config.train_mode + '/train_output.jpg')
                torchvision.utils.save_image(target[:,[2,1,0],:,:], './run-out/' + config.train_mode + '/GT_example.jpg')
                torchvision.utils.save_image(img_exposure[:,[2,1,0],:,:], './run-out/' + config.train_mode + '/input1.jpg')


            # 绘制损失曲线
            average_loss = epoch_loss / len(train_loader)

            epochloss = "%f"%average_loss
            range_ = "%f"%i
            epoch_ = "%f"%epoch
            list = [range_,epoch_,epochloss]
            data = pd.DataFrame([list])
            data.to_csv("data/loss-60col/loss.csv",mode="a",header=False,index=False)
            # 将平均损失值添加到列表中
            loss_curve.append(average_loss)
            # saver.save_model(w_net, average_loss, epoch)

            plt.plot(loss_curve)
            plt.title('Training Loss Curve')
            plt.xlabel('epoch')
            plt.ylabel('Loss')
            plt.tight_layout()  # 确保标签和标题等内容不会被图像边缘截断
            # 指定保存图像的路径
            save_path = f'data/loss-60col/loss_curve_{i}.png'
            # 保存图像到文件夹中
            plt.savefig(save_path)
            plt.close()  # 关闭图像，释放资源
        # plt.show()


if __name__ == "__main__":
    # 创建一个ArgumentParser对象，用于解析命令行参数
    parser = argparse.ArgumentParser()

    # 输入参数设置
    # 然后向该对象中添加要关注的命令行参数和选项，每一个add_argument方法对应一个要关注的参数或选项
    parser.add_argument('--train_mode', type=str, default='exp2')
    # # 指定低光照图像的路径，默认值为"data/train_data/"
    # parser.add_argument('--exposure_images_path', type=str, default="G:/light/dataset/MultiExposure_dataset/training/Patches/INPUT_IMAGES",
    #                     help='The path of input images')
    # parser.add_argument('--normal_images_path', type=str, default="G:/light/dataset/MultiExposure_dataset/training/Patches/GT_IMAGES",
    #                     help='The path of gt images')
    parser.add_argument('--exposure_images_path', type=str,
                        default="G:/light/dataset/multiexposure-people/training/Patchs/input",
                        help='The path of input images')
    parser.add_argument('--normal_images_path', type=str,
                        default="G:/light/dataset/multiexposure-people/training/Patchs/gt",
                        help='The path of gt images')

    parser.add_argument('--exposure_suffixes', nargs='*', default="['_P1', '_P1.5', '_0', '_N1.5', '_N1']")
    # 于指定学习率的值，默认值为0.0001
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--D_lr', type=float, default=1e-5)
    # 指定权重衰减的值，默认值为0.0001
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    # 指定梯度裁剪的阈值，默认值为0.1
    # parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    # parser.add_argument('--sizelist', type=list, default=[128, 256, 512])
    parser.add_argument('--sizelist', type=list, default=[256,384,512])
    # 指定训练的轮数，默认值为200
    parser.add_argument('--num_epochs_list', type=list, default=[40, 50, 90])
    # parser.add_argument('--num_epochs_list', type=list, default=[8,6,10])
    parser.add_argument('--start', type=list, default=[0, 0], help='[stage_start,epoch_start]')
    parser.add_argument('--train_batch_size_list', type=list, default=[24,10,6])
    # parser.add_argument('--train_batch_size_list', type=list, default=[96, 30, 6])
    # 指定验证批次的大小，默认值为4
    parser.add_argument('--val_batch_size', type=int, default=8)
    # 指定数据加载的线程数，默认值为4
    parser.add_argument('--num_workers', type=int, default=4)
    # 指定训练过程中的显示间隔，默认值为10
    parser.add_argument('--display_iter', type=int, default=20)
    # 指定保存模型的间隔
    parser.add_argument('--snapshot_iter', type=int, default=1)
    # 指定保存模型的文件夹路径
    parser.add_argument('--snapshots_folder', type=str, default="snapshots-nod/")
    # parser.add_argument('--G_snapshots_folder', type=str, default="snapshots/G/")
    parser.add_argument('--D_snapshots_folder', type=str, default="snapshots-nod/D/")

    # parser.add_argument('--loss_folder', type=str, default="data/loss-att/")
    # 指定保存模型的文件夹路径
    # parser.add_argument('--best_snapshots_folder', type=str, default="best/")

    parser.add_argument('--use_advloss', type=bool, default=False)
    # 指定是否加载预训练模型
    parser.add_argument('--load_pretrain', type=bool, default=True)
    # 指定预训练模型的文件路径，默认值为"snapshots/Epoch99.pth
    parser.add_argument('--pretrain_dir', type=str, default="snapshots-nod/512_epoch53.pth")
    # parser.add_argument('--D_pretrain_dir', type=str, default="snapshots-nod/D/384Dnet_epoch29.pth")


    # 调用parse_args()方法进行解析，解析成功之后即可使用
    # config通过解析命令行参数获得的一个对象。它包含了在命令行中指定的各种训练参数和配置选项的值
    config = parser.parse_args()

    loss_file_path = "data/loss/loss.csv"
    if not os.path.isfile(loss_file_path):
        # 如果文件不存在，创建一个新的DataFrame并为其添加表头
        df_headers = pd.DataFrame(columns=["range", "epoch", "epoch loss"])
        df_headers.to_csv(loss_file_path, mode='w', header=True, index=False)

    # 检查是否存在snapshots_folder目录，如果不存在，则创建snapshots_folder目录
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
        if not os.path.exists(config.D_snapshots_folder):
            os.mkdir(config.D_snapshots_folder)
    if not os.path.exists('./run-out/' + config.train_mode):
        os.makedirs('./run-out/' + config.train_mode)
    # 调用train函数，并将config对象作为参数传递，开始训练过程
    # 在调用train(config)时，config对象作为参数传递给train函数，以便在训练过程中使用这些配置选项和参数值。
    # 在train函数中，可以通过访问config对象的属性来获取相应的参数值，从而进行训练过程的设置和操作
    train(config)
