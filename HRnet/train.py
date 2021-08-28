#coding=utf-8
from collections import defaultdict
import torch.nn.functional as F
from loss import calc_loss
import time
import torch
from torch.utils.data import Dataset, DataLoader
from utils.lr_scheduler import adjust_learning_rate_poly
from utils.label2color import label_img_to_color, diff_label_img_to_color
from data import GaofenTrain, GaofenVal

from tqdm import tqdm
from evaluate import Evaluator
import numpy as np
import argparse

import cv2
cv2.setNumThreads(1)
import os
from tensorboardX import SummaryWriter
from model.seg_hrnet import hrnet18
from ipdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/media/yr/新加卷/syz/gaofen')               # 根目录
parser.add_argument('--train_list_path', default='/media/yr/新加卷/syz/gaofen/makeData/train.txt')                   # 训练集id路径
parser.add_argument('--val_list_path', default='/media/yr/新加卷/syz/gaofen/makeData/val.txt')                       # 验证集
parser.add_argument('--test_list_path', default='/media/yr/新加卷/syz/gaofen/makeData/test.txt')                     # 测试
parser.add_argument('--backbone', default='resnest50', type=str,help='xception|resnet|resnest101|resnest200|resnest50|resnest26')
parser.add_argument('--n_cls', default=2, type=int)                                    # 类别
parser.add_argument('--batchsize', default=4, type=int)                                # batchsize
parser.add_argument('--lr', default=0.01, type=float)                                  # 学习率
parser.add_argument('--num_epochs', default=170, type=int)                             # epoch
parser.add_argument('--warmup', default=100, type=int)
parser.add_argument('--multiplier', default=100, type=int)
parser.add_argument('--eta_min', default=0.0005, type=float)
parser.add_argument('--num_workers', default=8, type=int)                              # 线程
parser.add_argument('--decay_rate', default=0.8, type=float)
parser.add_argument('--decay_epoch', default=200, type=int)                            # 学习率衰减设置
parser.add_argument('--vis_frequency', default=1, type=int)                           # 验证可视化的频率
parser.add_argument('--save_path', default='./results')                                # 保存路径
parser.add_argument('--gpu-id', default='0,1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')                             # GPU
parser.add_argument('--is_resume', default=False, type=bool)                           # 是否重新训练模型
parser.add_argument('--resume', default='', type=str, help='./results/checkpoint.pth') # 参数保存路径
args = parser.parse_args()

folder_path = '/backbone={}/warmup={}_lr={}_multiplier={}_eta_min={}_num_epochs={}_batchsize={}'.format(args.backbone, args.warmup, args.lr, args.multiplier, args.eta_min, args.num_epochs,args.batchsize)
isExists = os.path.exists(args.save_path + folder_path)              # 判断文件是否存在
if not isExists:
    os.makedirs(args.save_path + folder_path)                        # 创建文件夹
isExists = os.path.exists(args.save_path +folder_path+'/vis')        # 创建 vis文件夹
if not isExists: os.makedirs(args.save_path + folder_path+'/vis')

isExists = os.path.exists(args.save_path +folder_path+'/log')        # 创建 log文件夹
if not isExists: os.makedirs(args.save_path + folder_path+'/log')

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 显卡管理

def train_model():
    F_txt = open('./opt_results.txt', 'w')  # 创建保存结果文本
    evaluator = Evaluator(args.n_cls)       # 指标类实例化
    writer = SummaryWriter(args.save_path + folder_path + '/log')  # 日志，记录训练过程
    model = hrnet18(pretrained=True).to(device)                 # 实例化模型
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)  # 随机梯度下降优化器
    # 参数初始化
    best_miou = 0.    # miou初始参数
    best_fwiou = 0.
    best_AA = 0.
    best_OA = 0.
    best_loss = 0.
    lr = args.lr
    epoch_index = 0

    if args.is_resume:  # 用于恢复训练
        args.resume = args.save_path + folder_path + '/checkpoint_fwiou.pth'
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            epoch_index = checkpoint['epoch']
            best_miou = checkpoint['miou']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            F_txt.write("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']) + '\n')
            # print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), file=F_txt)
        else:
            print('EORRO: No such file!!!!!')

    train_root = args.root                     # 训练的根路径
    train_list_path = args.train_list_path     # 文件名路径

    val_root = args.root
    val_list_path = args.val_list_path
    # 实例化dataloader
    dataloaders = {
        "train": DataLoader(GaofenTrain(train_root, train_list_path), batch_size=args.batchsize,
                            shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True),
        "val": DataLoader(GaofenVal(val_root, val_list_path), batch_size=args.batchsize,
                          num_workers=args.num_workers, pin_memory=True)
    }

    evaluator.reset()     # 设置2*2的0矩阵
    print('config: ' + folder_path)   # folder_path (包含训练参数)
    print('config: ' + folder_path, file=F_txt,flush=True)  # 将打印的内容存入F_txt中，flush=True刷新内容，新的打印内容会持续存入，这里写入的是参数

    for epoch in range(epoch_index, args.num_epochs):       # 开始迭代150代
        print('Epoch [{}]/[{}] lr={:6f}'.format(epoch + 1, args.num_epochs, lr))
        print('Epoch [{}]/[{}] lr={:4f}'.format(epoch + 1, args.num_epochs, lr), file=F_txt, flush=True) # 这里写入训练过程
        since = time.time()
        # 训练一代、验证一代
        for phase in ['train', 'val']:
            evaluator.reset()
            if phase == 'train':
                model.train()   # dropout层会按照设置好的失活概率进行失活，batchnorm会继续计算数据的均值和方差等参数并在每个batch size之间不断更新
            else:
                model.eval()    # eval主要是用来影响网络中的dropout层和batchnorm层的行为

            metrics = defaultdict(float)     # 默认字典
            epoch_samples = 0

            for i, (inputs, labels, edge, _, datafiles) in enumerate(tqdm(dataloaders[phase], ncols=50)):
                inputs = inputs.to(device)   # img
                edge = edge.to(device, dtype=torch.float)    # edg
                labels = labels.to(device, dtype=torch.long) # labels

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):   # 设置梯度反传，默认为false
                    outputs = model(inputs)   # outputs[1].shape----torch.Size([4, 2, 321, 321])  outputs[0].shape----torch.Size([4, 2, 81, 81])
                    outputs[1] = F.interpolate(input=outputs[1], size=(labels.shape[1], labels.shape[2]), mode='bilinear', align_corners=True)   # 输出双线性插值采样为label大小
                    loss = calc_loss(outputs, labels, edge, metrics)    # 损失
                    pred = outputs[1].data.cpu().numpy()      # 模型输出
                    pred = np.argmax(pred, axis=1)            # 输出预测标签
                    labels = labels.data.cpu().numpy()     # 标签
                    evaluator.add_batch(labels, pred)  # 评估

                    if phase == 'val' and (epoch + 1) % args.vis_frequency == 0 and inputs.shape[0] == args.batchsize:  # 验证每隔30代
                        for k in range(args.batchsize // 2):   # k = 0, 1
                            name = datafiles['name'][k][:-4]   # 当前batch图片名
                            writer.add_image('{}/img'.format(name), cv2.cvtColor(cv2.imread(datafiles["img"][k], cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB), global_step=int((epoch + 1)), dataformats='HWC')
                            # 图片格式转换（RGB）,添加到日志中
                            writer.add_image('{}/gt'.format(name), label_img_to_color(labels[k])[:, :, ::-1],global_step=int((epoch + 1)), dataformats='HWC')
                            # gt格式转换保存于日志中
                            pred_label_img = pred.astype(np.uint8)[k]    # batch_size中的pred的第一张图（512，512）
                            # 预测图（pred）
                            pred_label_img_color = label_img_to_color(pred_label_img)   # (512, 512, 3)
                            # 预测图
                            writer.add_image('{}/mask'.format(name), pred_label_img_color[:, :, ::-1],global_step=int((epoch + 1)), dataformats='HWC')
                            # mask 标签转化为三通道，代表预测标签
                            softmax_pred = F.softmax(outputs[1][k],dim=0)      # 上采样后的预测图通道维度softmax  torch.Size([2, 512, 512])
                            softmax_pred_np = softmax_pred.data.cpu().numpy()  # torch.Size([2, 512, 512])
                            probility = softmax_pred_np[0]
                            probility = probility*255          # 概率值乘255
                            probility = probility.astype(np.uint8)
                            probility = cv2.applyColorMap(probility,cv2.COLORMAP_HOT)   # 伪彩图  可以显示网络的关注点
                            writer.add_image('{}/prob'.format(name),cv2.cvtColor(probility,cv2.COLOR_BGR2RGB),global_step=int((epoch+1)),dataformats='HWC')
                            # 差分图
                            diff_img = np.ones((pred_label_img.shape[0], pred_label_img.shape[1]), dtype=np.int32)*255   # 制造全1
                            mask = (labels[k] != pred_label_img)          # 得到gt不等于1的mask
                            diff_img[mask] = labels[k][mask]
                            diff_img_color = diff_label_img_to_color(diff_img)  #  得到diff图
                            writer.add_image('{}/different_image'.format(name), diff_img_color[:, :, ::-1], global_step=int((epoch + 1)), dataformats='HWC')
                    if phase == 'train':
                        loss.backward()   # 反向传播
                        optimizer.step()  # 梯度更新
                        adjust_learning_rate_poly(args.lr,optimizer, epoch * len(dataloaders['train']) + i,args.num_epochs * len(dataloaders['train']))
                        lr = optimizer.param_groups[0]['lr']
                        writer.add_scalar('lr', lr, global_step=epoch * len(dataloaders['train']) + i)

                epoch_samples += 1
            epoch_loss = metrics['loss'] / epoch_samples    # 一个epoch的平均损失
            ce_loss = metrics['ce_loss'] / epoch_samples
            ls_loss = metrics['ls_loss'] / epoch_samples
            fwiou = evaluator.Frequency_Weighted_Intersection_over_Union()
            miou = evaluator.Mean_Intersection_over_Union()
            AA = evaluator.Pixel_Accuracy_Class()
            OA = evaluator.Pixel_Accuracy()
            if phase == 'val':
                miou_mat = evaluator.Mean_Intersection_over_Union_test()
                writer.add_scalar('val/val_loss', epoch_loss, global_step=epoch)   # 数据标识符，要保存的数值，全局步值
                writer.add_scalar('val/ce_loss', ce_loss, global_step=epoch)
                writer.add_scalar('val/ls_loss', ls_loss, global_step=epoch)
                writer.add_scalar('val/val_fwiou', fwiou, global_step=epoch)
                writer.add_scalar('val/val_miou', miou, global_step=epoch)
                for index in range(args.n_cls):
                    writer.add_scalar('class/{}'.format(index + 1), miou_mat[index], global_step=epoch)
                    print('[val]------fwiou: {:4f}, miou: {:4f}, OA:{:4f}, AA: {:4f}, loss: {:4f}'.format(fwiou, miou, OA, AA, epoch_loss))  # 打印
                    # 写入
                    print('[val]------fwiou: {:4f}, miou: {:4f}, OA:{:4f}, AA: {:4f}, loss: {:4f}'.format(fwiou, miou, OA, AA, epoch_loss),file=F_txt, flush=True)
            if phase == 'train':
                writer.add_scalar('train/train_loss', epoch_loss, global_step=epoch)
                writer.add_scalar('train/ce_loss', ce_loss, global_step=epoch)
                writer.add_scalar('train/ls_loss', ls_loss, global_step=epoch)
                writer.add_scalar('train/train_fwiou', fwiou, global_step=epoch)
                writer.add_scalar('train/train_miou', miou, global_step=epoch)
                print(
                    '[train]------fwiou: {:4f}, miou: {:4f}, OA: {:4f}, AA: {:4f}, loss: {:4f}'.format(fwiou, miou, OA, AA, epoch_loss))
                # 下面的print是写入
                print(
                    '[train]------fwiou: {:4f}, miou: {:4f}, OA: {:4f}, AA: {:4f}, loss: {:4f}'.format(fwiou, miou, OA, AA, epoch_loss), file=F_txt, flush=True)
            if phase == 'val' and fwiou > best_fwiou:
                print("\33[91msaving best model miou\33[0m")
                print("saving best model fwiou", file=F_txt, flush=True)
                best_miou = miou
                best_fwiou = fwiou
                best_OA = OA
                best_AA = AA
                best_loss = epoch_loss
                torch.save({
                    'name': 'resnest50_lovasz_edge_rotate',
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_miou': best_miou
                }, args.save_path + folder_path + '/model_best.pth')

                torch.save({
                    'optimizer': optimizer.state_dict(),
                }, args.save_path + folder_path + '/optimizer.pth')

        time_elapsed = time.time() - since                         # 一个epoch运行时间
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))      # 打印运行时间
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=F_txt, flush=True)  # 记录日志

    print('[Best val]------fwiou: {:4f}; miou: {:4f}; OA: {:4f}; AA: {:4f}; loss: {:4f}'.format(best_fwiou, best_miou, best_OA, best_AA, best_loss))
    print('[Best val]------fwiou: {:4f}; miou: {:4f}; OA: {:4f}; AA: {:4f}; loss: {:4f}'.format(best_fwiou, best_miou, best_OA, best_AA, best_loss),file=F_txt,flush=True)
    F_txt.close()


if __name__ == '__main__':
    train_model()






