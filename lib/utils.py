import numpy as np
from math import *

# 返回narray中最接近给定值的值
def findNearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]

# 计算毛发统计信息
def analysisInf(width_count,offset=0):
    temp = []
    for width, nums in enumerate(width_count):
        for i in range(int(nums)):
            temp.append(width+offset)
    temp = np.array(temp)

    num = int(width_count.sum())
    if num>0:
        num = str(num)
        median = str(np.median(temp))
        mean = str(format(np.mean(temp), '.2f'))
        std = str(format(np.std(temp), '.2f'))
        mode = str(np.argmax(np.bincount(temp)))
    else:
        num = '0'
        median = '0'
        mean = '0'
        std = '0'
        mode = '0'

    return {'num':num, 'median':median, 'mean':mean, 'std':std, 'mode':mode}

# 扩充骨架点————权宜之计，待改
def midUpsample(joints):

    nums = len(joints)
    new_joints=[]

    if nums==2:
        x0 = joints[0][0]
        y0 = joints[0][1]
        x1 = joints[1][0]
        y1 = joints[1][1]
        new_joints.append([x0,y0])
        new_joints.append([x0 + (x1 - x0) / 4, y0 + (y1 - y0) / 4])
        new_joints.append([x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2])
        new_joints.append([x0 + (x1 - x0) * 0.75, y0 + (y1 - y0) * 0.75])
        new_joints.append([x1,y1])
    elif nums==3:
        x0 = joints[0][0]
        y0 = joints[0][1]
        x1 = joints[1][0]
        y1 = joints[1][1]
        x2 = joints[2][0]
        y2 = joints[2][1]
        new_joints.append([x0, y0])
        new_joints.append([x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2])
        new_joints.append([x1, y1])
        new_joints.append([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2])
        new_joints.append([x2, y2])
    elif nums == 4:
        x0 = joints[0][0]
        y0 = joints[0][1]
        x1 = joints[1][0]
        y1 = joints[1][1]
        x2 = joints[2][0]
        y2 = joints[2][1]
        x3 = joints[3][0]
        y3 = joints[3][1]
        new_joints.append([x0, y0])
        new_joints.append([x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2])
        new_joints.append([x1, y1])
        new_joints.append([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2])
        new_joints.append([x2, y2])
        new_joints.append([x2 + (x3 - x2) / 2, y2 + (y3 - y2) / 2])
        new_joints.append([x3, y3])
    else:
        new_joints = joints
    return new_joints

