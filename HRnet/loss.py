import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from collections import defaultdict
from torch.autograd import Variable
import itertools
from ipdb import set_trace


def isnan(x):
    return x != x

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = itertools.filterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n



class CrossEntropy(nn.Module):  # 交叉熵
    def __init__(self, ignore_label=255, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label          # 255
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction='none')    # 另外一种计算交叉熵的方式，CrossEntropyLoss()函数的主要是将softmax-log-NLLLoss合并到一块得到的结果。

    def _forward(self, score, target):
        # 只是为了确保预测与标签的大小一致
        ph, pw = score.size(2), score.size(3)       # 获得图像的h,w     (batchszie, c, h, w)
        h, w = target.size(1), target.size(2)       # 标签的 h ,w      (bathsize,h,w)
        if ph != h or pw != w:                      # 判断大小是否一致，不相等则进行采样操作（双线性插值）
            score = F.interpolate(input=score, size=(h, w), mode='bilinear', align_corners=True)

        loss = self.criterion(score, target)    # 交叉熵计算

        return loss

    def forward(self, score, target):

        hr_weights = [0.4, 1]
        assert len(hr_weights) == len(score)      # score==pre为列表 长度为2
        loss = hr_weights[0]*self._forward(score[0], target) + hr_weights[1]*self._forward(score[1], target)
        return loss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)          预测概率在（0，1）之间
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)                             类别在（0，class-1）
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:           # numel()函数：返回数组中元素的个数
        # 只有无效像素，梯度应为0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):    # resize为一维向量
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: 计算每幅图像而不是每批图像的损失
      ignore: 无效的类标签
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss




def calc_loss(pred, target, edge, metrics):    # 模型预测、标签、边缘、metrics
    edge_weight = 4.                           # 边缘权重
    criters_ce = CrossEntropy()                # 调用交叉熵
    loss_ce = criters_ce(pred,target)          # 交叉熵损失    torch.Size([4, 321, 321])
    loss_ls = lovasz_softmax(F.softmax(pred[1],dim=1),target)     # ocr的输出结果在通道上做softmax
    edge[edge == 0] = 1.                       # edge    torch.Size([4, 321, 321])
    edge[edge == 255] = edge_weight            # 边缘交界线从255变为4.
    loss_ce *= edge                            # 加大边缘权重
    loss_ce = loss_ce[loss_ce!=0].mean()       # 求均值
    loss = loss_ce + loss_ls
    metrics['loss'] += loss.data.cpu().numpy()
    metrics['ce_loss'] += loss_ce.data.cpu().numpy()
    metrics['ls_loss'] += loss_ls.data.cpu().numpy()
    return loss



class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]



if __name__ == '__main__':
    criter = BinaryDiceLoss()     # 计算二分类的一种损失
    target = torch.ones((4, 256, 256), dtype=torch.long)
    input = (torch.ones((4, 256, 256)) * 0.9)
    loss = criter(input, target)
    print(loss)

