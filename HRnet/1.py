import numpy as np
import os


# gt_image = np.array([[1, 1],[1, 0]])
# pre_image = np.array([[0, 1],[2, 1]])
# print(gt_image[0])
# print(gt_image[0][gt_image[0]>0])


# mask = (gt_image >= 0) & (gt_image < 2)
# print('mask:%s'%mask)
# label = 2 * gt_image[mask].astype('int') + pre_image[mask]
# print('gt_image[mask]%s:'%gt_image[mask])
# print('pre_image[mask]%s:'%pre_image[mask])
# print(label)

s = os.walk('/media/yr/新加卷/syz/gaofen/test_img/')
print([file for _,_,file in s])

