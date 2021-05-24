import numpy as np
from math import *
import cv2
import math
import time

from numba.typed import List, Dict

def imageMergeMask(img, mask, invert=False):
    # 掩模显示背景
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)

    cv2.imshow('img1_bg', img1_bg)
    cv2.waitKey(0)

    # 掩模显示前景
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
    cv2.imshow('img2_fg', img2_fg)
    cv2.waitKey(0)

    # 前背景图像叠加
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst

    cv2.imshow('res', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img1[0:rows, 0:cols] = dst



# 返回narray中最接近给定值的值
def findNearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]

# 计算毛发统计信息
def analysisInf(width_count,pixel_width,offset=0,area=1):
    temp = []
    for width, nums in enumerate(width_count):
        for i in range(int(nums)):
            temp.append(width+offset)
    temp = np.array(temp)

    num = int(width_count.sum())
    if num>0:
        
        num = str(num)
        median = str(np.median(temp)*pixel_width) 
        mean = str(format(np.mean(temp)*pixel_width, '.2f'))
        std = str(format(np.std(temp)*pixel_width, '.2f'))
        mode = str(format(np.argmax(np.bincount(temp))*pixel_width, '.2f'))
        density = str(format(float(num)/area, '.2f'))
        #density = str(float(num)/area)
    else:
        density = '0'
        num = '0'
        median = '0'
        mean = '0'
        std = '0'
        mode = '0'

    return {'num':num, 'median':median, 'mean':mean, 'std':std, 'mode':mode, 'density':density}

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
        new_joints.append([int(x0 + (x1 - x0) / 4), int(y0 + (y1 - y0) / 4)])
        new_joints.append([int(x0 + (x1 - x0) / 2), int(y0 + (y1 - y0) / 2)])
        new_joints.append([int(x0 + (x1 - x0) * 0.75), int(y0 + (y1 - y0) * 0.75)])
        new_joints.append([x1,y1])
    elif nums==3:
        x0 = joints[0][0]
        y0 = joints[0][1]
        x1 = joints[1][0]
        y1 = joints[1][1]
        x2 = joints[2][0]
        y2 = joints[2][1]
        new_joints.append([x0, y0])
        new_joints.append([int(x0 + (x1 - x0) / 2), int(y0 + (y1 - y0) / 2)])
        new_joints.append([x1, y1])
        new_joints.append([int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)])
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
        new_joints.append([int(x0 + (x1 - x0) / 2), int(y0 + (y1 - y0) / 2)])
        new_joints.append([x1, y1])
        new_joints.append([int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)])
        new_joints.append([x2, y2])
        new_joints.append([int(x2 + (x3 - x2) / 2), int(y2 + (y3 - y2) / 2)])
        new_joints.append([x3, y3])
    elif nums == 5 :
        x0 = joints[0][0]
        y0 = joints[0][1]
        x1 = joints[1][0]
        y1 = joints[1][1]
        x2 = joints[2][0]
        y2 = joints[2][1]
        x3 = joints[3][0]
        y3 = joints[3][1]
        x4 = joints[4][0]
        y4 = joints[4][1]
        new_joints.append([x0, y0])
        new_joints.append([int(x0 + (x1 - x0) / 2), int(y0 + (y1 - y0) / 2)])
        new_joints.append([x1, y1])
        new_joints.append([int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)])
        new_joints.append([x2, y2])
        new_joints.append([int(x2 + (x3 - x2) / 2), int(y2 + (y3 - y2) / 2)])
        new_joints.append([x3, y3])
        new_joints.append([int(x3 + (x4 - x3) / 2), int(y3 + (y4 - y3) / 2)])
        new_joints.append([x4, y4])
    elif nums == 6 :
        x0 = joints[0][0]
        y0 = joints[0][1]
        x1 = joints[1][0]
        y1 = joints[1][1]
        x2 = joints[2][0]
        y2 = joints[2][1]
        x3 = joints[3][0]
        y3 = joints[3][1]
        x4 = joints[4][0]
        y4 = joints[4][1]
        x5 = joints[5][0]
        y5 = joints[5][1]
        new_joints.append([x0, y0])
        new_joints.append([int(x0 + (x1 - x0) / 2), int(y0 + (y1 - y0) / 2)])
        new_joints.append([x1, y1])
        new_joints.append([int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)])
        new_joints.append([x2, y2])
        new_joints.append([int(x2 + (x3 - x2) / 2), int(y2 + (y3 - y2) / 2)])
        new_joints.append([x3, y3])
        new_joints.append([int(x3 + (x4 - x3) / 2), int(y3 + (y4 - y3) / 2)])
        new_joints.append([x4, y4])
        new_joints.append([int(x4 + (x5 - x4) / 2), int(y4 + (y5 - y4) / 2)])
        new_joints.append([x5, y5])
    else:
        new_joints = joints
    return new_joints



def calculate_dis(point1,point2):
    return (point2[0] - point1[0]) * (point2[0] - point1[0]) + (point2[1] - point1[1]) * (point2[1] - point1[1])



def overlap(box1, box2):
    # 判断两个矩形是否相交
    # 思路来源于:https://www.cnblogs.com/avril/archive/2013/04/01/2993875.html
    # 然后把思路写成了代码
    minx1, miny1, maxx1, maxy1 = box1
    minx2, miny2, maxx2, maxy2 = box2
    minx = max(minx1, minx2)
    miny = max(miny1, miny2)
    maxx = min(maxx1, maxx2)
    maxy = min(maxy1, maxy2)
    if minx > maxx or miny > maxy:
        return False
    else:
        return True


def magnetXY(binary, candidate):
    n, m = binary.shape

    final_res = List()
    for ansy, ansx in candidate:
        if (ansx < 0) or (ansx >= n) or (ansy < 0) or (ansy >= m):
            continue
        if (binary[ansx, ansy] == 0):
            tmp, tmpdist = (-1, -1), -1.0
            for dx in range(max(-10, -ansx), min(+11, n - ansx)):
                for dy in range(max(-10, -ansy), min(+11, m - ansy)):
                    ansx_, ansy_ = ansx + dx, ansy + dy
                    curdist = math.sqrt(dx ** 2 + dy ** 2)
                    if (binary[ansx_, ansy_] == 1) and ((tmpdist < 0) or (curdist < tmpdist)):
                        tmp, tmpdist = (ansx_, ansy_), curdist
            if (tmpdist > 0):
                final_res.append(tmp)
        else:
            final_res.append((ansy, ansx))

    return final_res