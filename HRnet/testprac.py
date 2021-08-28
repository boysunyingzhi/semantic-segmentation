#coding=utf-8
from collections import defaultdict
import torch.nn.functional as F
from loss import calc_loss
import time
import torch
from torch.utils.data import Dataset, DataLoader
from data import GaofenVal
from tqdm import tqdm
from  evaluate import Evaluator
import torch.nn as nn
from scipy import ndimage
import torch, cv2
import numpy as np
import sys
from torch.autograd import Variable
import os
from model.seg_hrnet import hrnet18
from math import ceil
from utils.label2color import label_img_to_color, diff_label_img_to_color
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--backbone', default='resnest50', type=str,help='xception|resnet|resnest101|resnest200|resnest50')

parser.add_argument('--batchsize', default=4, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--num_epochs', default=150, type=int)
parser.add_argument('--warmup', default=100, type=int)
parser.add_argument('--multiplier', default=100, type=int)
parser.add_argument('--eta_min', default=0.0005, type=float)
args = parser.parse_args()

folder_path = '/backbone={}/warmup={}_lr={}_multiplier={}_eta_min={}_num_epochs={}_batchsize={}'.format(args.backbone,args.warmup,args.lr, args.multiplier, args.eta_min, args.num_epochs,args.batchsize)

save_dir='./results' + folder_path+'/vis'
resume_path = './results'+folder_path+'/model_best.pth'
root = '/media/ws/新加卷/wy/dataset/data'
test_list_path = '/media/yr/新加卷/syz/gaofen/makeData/test.txt'


#HRNet
def inference(model, image, flip=True):      # 变换后的图像输入
    size = image.size()        # image的shape 1chw
    pred = model(image)[1]     # 预测图像    2*2hw

    pred = F.interpolate(
        input=pred, size=size[-2:],
        mode='bilinear', align_corners=True
    )                          # 对预测进行采样

    if flip:  # 是否翻转
        flip_img = image.cpu().numpy()[:, :, :, ::-1]   # 通道转换
        flip_output = model(torch.from_numpy(flip_img.copy()).cuda())  # 输入模型

        flip_output = F.interpolate(
            input=flip_output, size=size[-2:],
            mode='bilinear', align_corners=True
        )                                               # 模型输出采样

        flip_pred = flip_output.cpu().numpy().copy()    # 采样后作为预测结果
        flip_pred = torch.from_numpy(flip_pred[:, :, :, ::-1].copy()).cuda()
        pred += flip_pred
        pred = pred * 0.5
    return pred#.exp()

def multi_scale_aug(image, rand_scale=1):      # 模型输出的图片进行输入
    long_size = np.int(512 * rand_scale + 0.5)  # 尺寸变化
    h, w = image.shape[:2]                      # hwc
    if h > w:
        new_h = long_size     # 新的高度
        new_w = np.int(w * long_size / h + 0.5) # 新的长度
    else:
        new_w = long_size
        new_h = np.int(h * long_size / w + 0.5)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # resize
    return image

def multi_scale_inference(model, image, scales=[0.75, 1., 1.25], flip=False):   # 一张一张的输入
    batch, _, ori_height, ori_width = image.size()     # 模型输出图像的尺寸（裁剪、缩放）
    assert batch == 1, "only supporting batchsize 1."
    image = image.cpu().numpy()[0].transpose((1, 2, 0)).copy()  # hwc
    stride_h = np.int(512 * 1.0)                      # 需要修改为输入图片的尺寸，即原图尺寸
    stride_w = np.int(512 * 1.0)
    final_pred = torch.zeros([1, 2, ori_height, ori_width]).cuda()   # 最终预测图为这个尺寸（原图的尺寸）

    for scale in scales:   # 尺度变换
        new_img = multi_scale_aug(image=image,rand_scale=scale)  # 返回新尺度的图像 hwc
        height, width = new_img.shape[:-1]

        if scale <= 2.0:
            new_img = new_img.transpose((2, 0, 1))       # chw
            new_img = np.expand_dims(new_img, axis=0)    # 1chw
            new_img = torch.from_numpy(new_img).cuda()   # 变换后的图像输入模型
            preds = inference(model, new_img, flip)      # 变换后图像的预测结果
            preds = preds[:, :, 0:height, 0:width]       # 切片为原图大小
        else:
            new_h, new_w = new_img.shape[:-1]
            rows = np.int(np.ceil(1.0 * (new_h - 512) / stride_h)) + 1
            cols = np.int(np.ceil(1.0 * (new_w - 512) / stride_w)) + 1
            preds = torch.zeros([1, 2, new_h, new_w]).cuda()
            count = torch.zeros([1, 1, new_h, new_w]).cuda()

            for r in range(rows):
                for c in range(cols):
                    h0 = r * stride_h
                    w0 = c * stride_w
                    h1 = min(h0 + 512, new_h)
                    w1 = min(w0 + 512, new_w)
                    h0 = max(int(h1 - 512), 0)
                    w0 = max(int(w1 - 512), 0)
                    crop_img = new_img[h0:h1, w0:w1, :]
                    crop_img = crop_img.transpose((2, 0, 1))
                    crop_img = np.expand_dims(crop_img, axis=0)
                    crop_img = torch.from_numpy(crop_img).cuda()
                    pred = inference(model, crop_img, flip)
                    preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                    count[:, :, h0:h1, w0:w1] += 1
            preds = preds / count
            preds = preds[:, :, :height, :width]

        preds = F.interpolate(preds, (ori_height, ori_width),mode='bilinear', align_corners=True)  # 预测结果采样为原始图片的大小
        final_pred += preds   # 多尺度的叠加
    return final_pred



batchsize = 1
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
evaluator=Evaluator(2)
classes=['roat','others']

def test():
    phase = 'test'
    model = hrnet18(pretrained=False).cuda()  # 实例化模型
    print("=> loading checkpoint '{}'".format(resume_path))
    checkpoint = torch.load(resume_path)  # 加载训练权重
    print(checkpoint['epoch'],checkpoint['best_miou'])
    model.load_state_dict(checkpoint['state_dict'])  # 模型加载权重
    model.eval()

    TEST_DATA_DIRECTORY = root      # 根目录
    TEST_DATA_LIST_PATH = test_list_path  # 测试文本路径

    dataloaders = {'test':DataLoader(GaofenVal(TEST_DATA_DIRECTORY,TEST_DATA_LIST_PATH), batch_size=batchsize,num_workers=4)}
    # test  name
    samples = 0
    metrics = defaultdict(float)    # 字典
    for i, (inputs, labels, edge, _, datafiles) in enumerate(tqdm(dataloaders['test'])):
        inputs = inputs.to(device)
        labels = labels.to(device,dtype=torch.long)

        with torch.set_grad_enabled(phase == 'train'):
            outputs = multi_scale_inference(model, inputs, scales=[0.5, 0.75, 1, 1.25, 1.5])  # 0.5,1.,1.25,1.5,1.75,2.
            # 输出原图大小的预测图1chw
            name = datafiles['name'][0][:-4]                 # 文件名,去掉png
            img = cv2.imread(datafiles["img"][0], cv2.IMREAD_COLOR)  # 读取原图
            filename = os.path.join(save_dir, '{}.png'.format(name)) # 保存路径
            cv2.imwrite(filename, img)                               # 存储原图到该路径

            # 绘制预概率图像
            softmax_pred = F.softmax(outputs,dim=1)                  # 通道维度softmax
            softmax_pred_np = softmax_pred.data.cpu().numpy()        # 1*chw
            probility = softmax_pred_np[:, 0]

            name = datafiles['name'][0][:-4]  # 文件名
            filename = os.path.join(save_dir, 'pre_{}_probility.png'.format(name))  # 文件保存路径
            probility = probility[0] * 255
            probility = probility.astype(np.uint8)
            probility = cv2.applyColorMap(probility, cv2.COLORMAP_HOT)  # 使用为彩图
            cv2.imwrite(filename, probility)  # 保存图片

            outputs = torch.softmax(outputs, dim=1)  # 通道维度softmax
            outputs = outputs.data.cpu().numpy()  # (1,9,512,512)
            pred = np.ones((1, outputs.shape[2], outputs.shape[3]), dtype=np.uint8)
            pred[outputs[:, 0, :, :] >= 0.3] = 0  # 输出的road置零，其他为1

            # pred = outputs.data.cpu().numpy() #(1,9,1024,1024)
            # pred = np.argmax(pred, axis=1)
            labels = labels.data.cpu().numpy()
            evaluator.add_batch(labels, pred)  # # may be delete !!!  计算指标
        pred_label_img = pred.astype(np.uint8)[0]
        pred_label_img_color = label_img_to_color(pred_label_img)  # 转为彩色图
        name = datafiles['name'][0][:-4]
        filename = os.path.join(save_dir, 'pre_{}.png'.format(name))
        cv2.imwrite(filename, pred_label_img_color)  # 保存彩色图
        filename = os.path.join(save_dir, 'gt_{}.png'.format(name))
        cv2.imwrite(filename, label_img_to_color(labels[0]))  # 保存标签

        diff_img = np.ones((pred_label_img.shape[0], pred_label_img.shape[1]), dtype=np.int32) * 255
        mask = (labels[0] != pred_label_img)
        diff_img[mask] = labels[0][mask]
        filename = os.path.join(save_dir, 'diff_{}.png'.format(name))
        cv2.imwrite(filename, diff_label_img_to_color(diff_img))
    print('miou: {:4f}'.format(evaluator.Mean_Intersection_over_Union()))
    print('fwiou: {:4f}'.format(evaluator.Frequency_Weighted_Intersection_over_Union()))


if __name__ == '__main__':
    test()