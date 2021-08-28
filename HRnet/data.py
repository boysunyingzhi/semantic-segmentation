import cv2
import numpy as np
import os.path as osp
import random
from torch.utils import data
from scipy.ndimage.morphology import distance_transform_edt
from ipdb import set_trace

class GaofenTrain(data.Dataset):
    def __init__(self, root, list_path,  crop_size=(321, 321),
                 scale=True, mirror=True,rotation=True, bright=False, ignore_label=1, use_aug=True, network='resnet101'):
        self.root = root  # 文件根目录
        self.src_h = 512  # 图片高
        self.src_w = 512  # 宽
        self.list_path = list_path  # 文件目录
        self.crop_h, self.crop_w = crop_size  # 裁剪后的长宽
        self.bright = bright  #
        self.scale = scale  # 比例/尺度变化
        self.ignore_label = ignore_label  #
        self.is_mirror = mirror  # 翻转
        self.rotation = rotation  # 旋转
        self.use_aug = use_aug
        self.img_ids = [i_id.strip() for i_id in open(list_path)]  # 图片的id列表   train.txt
        self.files = []  # 文件列表
        self.network = network  # 特征提取网络

        for item in self.img_ids:
            image_path = '/media/yr/新加卷/syz/gaofen/trainData/image/'+item
            label_path = '/media/yr/新加卷/syz/gaofen/trainData/gt/'+item
            name = item
            img_file = image_path
            label_file = label_path
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
                "weight": 1
            })                                # 列表中存储字典，每个字典代表一个数据
        print('{} images are loaded!'.format(len(self.img_ids)))  # 打印添加到列表中图片数量

    def __len__(self):
        return len(self.files)

    def random_brightness(self,img):       # 亮度调节
        if random.random() < 0.5:
            return img
        self.shift_value = 10 #取自HRNet
        img = img.astype(np.float32)
        shift = random.randint(-self.shift_value, self.shift_value)   # 调节区间[-10,10]的整数
        img[:, :, :] += shift                                         # 图片整体加上一个随机值
        img = np.around(img)                                          # 取整
        img = np.clip(img, 0, 255).astype(np.uint8)                   # 把元素限制在0-255之间
        return img

    def generate_scale_label(self, image, label):                     # 尺度变化缩放大小，改变图像的分辨率
        f_scale = 0.5 + random.randint(0, 11) / 10.0 # [0.5, 1.5]
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)  # fx：沿水平轴缩放的比例因子
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST) # 将图片进行压缩
        return image, label

    def mask_to_onehot(self,mask, num_classes):
        '''
        将分割掩码(H,W)转换为(K,H,W)，其中最后的dim是一个热编码向量
        '''
        _mask = [mask == (i) for i in range(num_classes)]
        return np.array(_mask).astype(np.uint8)

    def onehot_to_binary_edges(self, mask, radius, num_classes):
        """
        Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)
        """
        if radius < 0:              # 边界的半径为radius
            return mask
        # We need to pad the borders for boundary conditions
        mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant',
                          constant_values=0)  # 对图片进行常数填充，防止卷积后导致输出图像缩小和图像边缘信息丢失
        edgemap = np.zeros(mask.shape[1:])    # 创建与图像相同大小的背景
        for i in range(num_classes):
            # ti qu lun kuo
            dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(
                1.0 - mask_pad[i, :])  # 用于距离转换，计算图像中非零点到最近背景点（即0）的距离
            # dist 为计算完距离后的mask
            dist = dist[1:-1, 1:-1]  # 实际图像的计算结果
            dist[dist > radius] = 0  # 距离大于radiu的位置像素值置为0
            edgemap += dist  # 加距离掩膜
        # edgemap = np.expand_dims(edgemap, axis=0)
        edgemap = (edgemap > 0).astype(np.uint8) * 255  # 对edgemap中值大于0的*255
        return edgemap



    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"],cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], 0)
        # 旋转90/180/270
        if self.rotation and random.random() > 0.5:
            angel = np.random.randint(1, 4)    # 1-3随机数
            M = cv2.getRotationMatrix2D(((self.src_h - 1) / 2., (self.src_w - 1) / 2.), 90*angel, 1) #第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
            image = cv2.warpAffine(image, M, (self.src_h, self.src_w), flags=cv2.INTER_LINEAR)  #仿射变换，线性插值
            label = cv2.warpAffine(label, M, (self.src_h, self.src_w), flags=cv2.INTER_NEAREST, borderValue=self.ignore_label)
                                                                                                # label 边界值填充为1
        # 旋转-30-30
        if self.rotation and random.random() > 0.5:
            angel = np.random.randint(-30,30)
            M = cv2.getRotationMatrix2D(((self.src_h - 1) / 2., (self.src_w - 1) / 2.), angel, 1)
            image = cv2.warpAffine(image, M, (self.src_h, self.src_w), flags=cv2.INTER_LINEAR)
            label = cv2.warpAffine(label, M, (self.src_h, self.src_w), flags=cv2.INTER_NEAREST, borderValue=self.ignore_label)
        size = image.shape  # 旋转之后图片的大小
        if self.scale: #尺度变化缩放
            image, label = self.generate_scale_label(image, label)
        if self.bright: #亮度变化
            image = self.random_brightness(image)
        image = np.asarray(image, np.float32)  # 将结构数据转化为多维数组
        image = image[:, :, ::-1]              # 原为GBR，转RGB通道
        mean = (0.14804721074317062, 0.14804721074317062, 0.14804721074317062)
        std = (0.061486173222318835, 0.061486173222318835, 0.061486173222318835)
        image /= 255.                 # 标准化
        image -= mean
        image /= std
        label = np.asarray(label, np.float32)
        label /= 255.                 # 标签归一化

        img_h, img_w = label.shape  # gt图片大小
        pad_h = max(self.crop_h - img_h, 0)               # 裁剪高 - 标签高
        pad_w = max(self.crop_w - img_w, 0)               # 裁剪宽 - 标签宽
        # 为图像添加边框
        if pad_h > 0 or pad_w > 0:                        # 界框部分设置为0
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))                # 图片设置边界框，就像一个相框一样的东西，cv2.BORDER_CONSTANT添加像素值为常数
                                                                               # 如果borderType为cv2.BORDER_CONSTANT时需要填充的常数值
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))  #边界填充的是ignore
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape  # 添加边框后标签高、宽

        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        image = image.transpose((2, 0, 1))      # 3*H*W  通道放在前面

        if self.is_mirror: #水平/垂直翻转
            flip1 = np.random.choice(2) * 2 - 1   # 取值为1或者-1
            label = label[:, ::flip1]
            flip2 = np.random.choice(2) * 2 - 1
            image = image[:,::flip2, :]           # 是否垂直翻转
            label = label[::flip2,:]
        oneHot_label = self.mask_to_onehot(label,2) #edge=255,background=0
        edge = self.onehot_to_binary_edges(oneHot_label,2,2)
        # 消去图像边缘
        edge[:2, :] = 0
        edge[-2:, :] = 0
        edge[:, :2] = 0
        edge[:, -2:] = 0
        return image.copy(), label.copy(), edge, np.array(size), datafiles


class GaofenVal(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321),
                 scale=False, mirror=False, ignore_label=255, use_aug=True, network="renset101"):
        self.root = root                                               # 根目录
        self.list_path = list_path                                     # 文件目录
        self.crop_h, self.crop_w = crop_size                           # 裁剪尺寸
        self.scale = scale                                             # 比例/尺寸
        self.ignore_label = ignore_label                               #
        self.is_mirror = mirror                                        # 翻转
        self.use_aug = use_aug
        self.img_ids = [i_id.strip() for i_id in open(list_path)]      # 图片id
        self.files = []
        self.network = network
        for item in self.img_ids:
            img_file = '/media/yr/新加卷/syz/gaofen/trainData/image/' + item
            label_file = '/media/yr/新加卷/syz/gaofen/trainData/gt/' + item
            name = item
            self.files.append({
                "img": img_file,
                "label": label_file,
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
        label = cv2.imread(datafiles["label"], 0)                    # 加载gt

        size = image.shape                                          # 图像尺寸

        image = np.asarray(image, np.float32)                       # 转多维数组
        image = image[:, :, ::-1]                                   # 转换通道顺序
        mean = (0.14804721074317062, 0.14804721074317062, 0.14804721074317062)
        std = (0.061486173222318835, 0.061486173222318835, 0.061486173222318835)
        image /= 255.                                               # 图片归一化
        image -= mean                                               # 图片标准化
        image /= std

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)/255
        image = image.transpose((2, 0, 1)) # 3XHXW
        return image.copy(), label.copy(),label.copy(), np.array(size), datafiles

