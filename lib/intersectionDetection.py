#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
from matplotlib import pyplot as plt
#from scipy.io import savemat
from time import time
from numba import jit

r = 20

tmp = np.arange(1, r + 1)[np.newaxis, :]
degree = (np.arange(0, 360)/360. * 2 * np.pi)[:, np.newaxis]
dx = np.floor(tmp * np.cos(degree)).astype(np.int32)
dx_ = np.ceil(tmp * np.cos(degree)).astype(np.int32)
dy = np.floor(tmp * np.sin(degree)).astype(np.int32)
dy_ = np.ceil(tmp * np.sin(degree)).astype(np.int32)
del tmp

@jit(nopython = True)
def _check(binary, x, y):
    xx, yy = int(np.floor(x)), int(np.floor(y))
    xx_, yy_ = int(np.ceil(x)), int(np.ceil(y))
    return binary[xx, yy] or binary[xx_, yy] or binary[xx, yy_] or binary[xx_, yy_]

@jit(nopython = True)
def _is_intersection(binary, x, y):
        
    # if (binary[x, y] == 0) or not ((binary[x-1, y] & binary[x, y-1] & binary[x+1, y] & binary[x, y+1]) == 1):
    #     return None

    # values = np.max( (binary[x + dx, y + dy], binary[x + dx_, y + dy], 
    #                   binary[x + dx, y + dy_], binary[x + dx_, y+dy_]), axis = 0 )
    # flag = values.sum(axis = 1) == r
    # s = (flag[1:] != flag[:-1]).sum() + (flag[0] != flag[-1])

    s = 0
    lastflag = -1
    for degree in range(361):
        rad = degree / 360 * 2 * np.pi
        flag = 1
        for k in range(1, r+1):
            if not _check(binary, x + np.cos(rad) * k, y + np.sin(rad) * k):
                flag = 0
                break
        s += (flag != lastflag)
        lastflag = flag
    s -= 1
    
    return s > 4

def find_intersection(binary, intersection = None, r = 10):
    '''
    Find all of the intersections of the hairs in a given binary image.
    
    Args:
    ------
    binary : 2D numpy.array. A binary image.
    r : int, optional. Should be in [1, 30]. The radius of the neibourhood circle.
   
    Returns:
    ------
    endpoints : 2D numpy.array, [(x1, y1), (x2, y2), ...] Each row represents an end point. 
    '''
    
    n, m = binary.shape

    shifted = np.zeros((n + 2 * r + 1, m + 2 * r + 1), dtype = np.uint8)
    shifted[r:r+n, r:r+m] = binary
    
    if (intersection is None):
        res = np.zeros((n, m), dtype = bool)
        for i in range(r, r+n):
            for j in range(r, r+m):
                if (_is_intersection(shifted, i, j)):
                    res[(i - r, j - r)] = 1
        return res
    else:
        for i in range(r, r+n):
            for j in range(r, r+m):
                if intersection[i - r, j - r] and not (_is_intersection(shifted, i, j)):
                    intersection[(i - r, j - r)] = 0
        return intersection

def is_intersection(binary, candidate):
    n, m = binary.shape

    shifted = np.zeros((n + 2 * r + 1, m + 2 * r + 1), dtype = np.uint8)
    shifted[r:r+n, r:r+m] = binary

    res = np.zeros( len(candidate), dtype = bool )

    for i in range(len(candidate)):
        x, y = candidate[i] + r
        res[i] = (_is_intersection(shifted, x, y))
    
    return res

def intersectionDetection(img_binary, intersection = None):
    """
        输入二值图，输出一个mask，表示哪些点位于交叉附近。
        Args:
            img_binary (ndarray) : 单通道二值图
        Returns:
            mask (ndarray) 
    """
    return find_intersection(img_binary, intersection = intersection)

if (__name__ == "__main__"):
    
    img = cv2.imread('binary.jpg', cv2.IMREAD_GRAYSCALE)
    binary = (img > 127).astype(np.uint8)
    
    t0 = time()
    ans = intersectionDetection(binary)
    print(time() - t0, "s")

    binary[binary > 0] = 255
    binary[ans] = 127
    plt.imshow(binary, cmap = plt.cm.gray)
    plt.savefig("intersection.pdf", dpi = 1200)
    
    #savemat("end_points.mat", {"end_points" : ans})





