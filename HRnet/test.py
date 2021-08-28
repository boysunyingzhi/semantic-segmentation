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
import cv2
import numpy as np
import os.path as osp
import random
from torch.utils import data
from scipy.ndimage.morphology import distance_transform_edt
from ipdb import set_trace
import os

class Gaofentest(data.Dataset):
    def __init__(self, input_path, max_iters=None, crop_size=(321, 321),scale=False, mirror=False, ignore_label=255, use_aug=True, network="renset101"):
        self.input_path = input_path                                   # 文件目录
        self.crop_h, self.crop_w = crop_size                           # 裁剪尺寸
        self.scale = scale                                             # 比例/尺寸
        self.ignore_label = ignore_label                               #
        self.is_mirror = mirror                                        # 翻转
        self.use_aug = use_aug
        self.img_file = os.walk(self.input_path)
        self.img_ids = [file for _, _, file in self.img_file]   # 图片id
        self.files = []
        self.network = network


        for item in self.img_ids[0]:
            img_file = input_path + item
            name = item
            self.files.append({
                "img": img_file,
                "name": name,
                "weight": 1
            })
        self.id_to_trainid = {}

        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]                               # 文件迭代索引
        image = cv2.imread(datafiles["img"])                        # 加载图像
        size = image.shape                                          # 图像尺寸

        image = np.asarray(image, np.float32)                       # 转多维数组
        image = image[:, :, ::-1]                                   # 转换通道顺序
        mean = (0.14804721074317062, 0.14804721074317062, 0.14804721074317062)
        std = (0.061486173222318835, 0.061486173222318835, 0.061486173222318835)
        image /= 255.                                               # 图片归一化
        image -= mean                                               # 图片标准化
        image /= std

        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1)) # 3XHXW
        return image.copy(), np.array(size), datafiles


backbone = 'resnest50'
batchsize = 4
lr = 0.01
num_epochs = 150
warmup = 100
multiplier = 100
eta_min = 0.0005

folder_path = '/backbone={}/warmup={}_lr={}_multiplier={}_eta_min={}_num_epochs={}_batchsize={}'.format(backbone,warmup,lr, multiplier, eta_min, num_epochs,batchsize)

save_dir='./results' + folder_path+'/vis'
resume_path = './results'+folder_path+'/model_best.pth'
batchsize = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def test(input_path, output_path):
    model = hrnet18(pretrained=False).cuda()  # 实例化模型
    print("=> loading checkpoint '{}'".format(resume_path))
    checkpoint = torch.load(resume_path)  # 加载训练权重
    print(checkpoint['epoch'],checkpoint['best_miou'])
    model.load_state_dict(checkpoint['state_dict'])  # 模型加载权重
    model.eval()

    dataloaders = {'test': DataLoader(Gaofentest(input_path), batch_size=batchsize, num_workers=4)}

    for i, (inputs, sizes, datafiles) in enumerate(tqdm(dataloaders['test'])):
        inputs = inputs.to(device)
        pred = model(inputs)[1]
        pred = F.interpolate(pred, size=(sizes[0][0], sizes[0][1]), mode='bilinear', align_corners=True)
        pred = pred.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        pred_label_img = pred.astype(np.uint8)
        pred_label_img = pred_label_img.squeeze(0)
        pred_label_img_color = label_img_to_color(pred_label_img)
        name = datafiles['name'][0][:-4]
        filename = os.path.join(output_path, 'pre_{}.png'.format(name))
        cv2.imwrite(filename, pred_label_img_color)

if __name__ == '__main__':
    test(input_path='/media/yr/新加卷/syz/gaofen/test_img/', output_path='/media/yr/新加卷/syz/gaofen/test_img/')