import numpy as np

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class                                 # 初始化分类的类别数
        self.confusion_matrix = np.zeros((self.num_class,)*2)      # 创建(2,2)的矩阵

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()   # 对角线元素之和/矩阵所有元素之和
        # np.diag()输出对角线元素（2dim以上）
        return Acc

    def Pixel_Accuracy_Class(self):
        # print(self.confusion_matrix.sum(axis=1))
        # print(self.confusion_matrix.sum())
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        # numpy.nanmean()函数可用于计算忽略NaN值的数组平均值。如果数组具有NaN值，我们可以找出不受NaN值影响的均值
        return MIoU
    def Mean_Intersection_over_Union_test(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):       # 标签，预测图
        mask = (gt_image >= 0) & (gt_image < self.num_class)    # &位运算 得到的mask   二进制与运算
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]  # pre_image[mask]得到一维向量，类似于矩阵拉平
        count = np.bincount(label, minlength=self.num_class**2)    # 该函数就是将原来数组 x 中的每一项出现的频次记录下来,0出现的频次，1出现的频次，最小长度为4
        confusion_matrix = count.reshape(self.num_class, self.num_class) # 最终结果：0代表TF 1:FP 2:FN 3:TP
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):     # 输入标签和预测结果
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)    # 创建混淆矩阵