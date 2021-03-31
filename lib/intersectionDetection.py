#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
from matplotlib import pyplot as plt
#from scipy.io import savemat

def find_intersection(binary, r = 10):
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
    
    tmp = np.arange(1, r + 1)[np.newaxis, :]
    degree = (np.arange(0, 360)/360. * 2 * np.pi)[:, np.newaxis]
    dx = np.floor(tmp * np.cos(degree)).astype(np.int32)
    dx_ = np.ceil(tmp * np.cos(degree)).astype(np.int32)
    dy = np.floor(tmp * np.sin(degree)).astype(np.int32)
    dy_ = np.ceil(tmp * np.sin(degree)).astype(np.int32)
    del tmp
    
    def _is_intersection(binary, x, y):
        
        if (binary[x, y] == 0) or not ((binary[x-1, y] & binary[x, y-1] & binary[x+1, y] & binary[x, y+1]) == 1):
            return None

        values = np.max( (binary[x + dx, y + dy], binary[x + dx_, y + dy], 
                          binary[x + dx, y + dy_], binary[x + dx_, y+dy_]), axis = 0 )
        flag = values.sum(axis = 1) == r
        s = (flag[1:] != flag[:-1]).sum() + (flag[0] != flag[-1])

        return s > 4

    shifted = np.zeros((n + 2 * r + 1, m + 2 * r + 1), dtype = np.uint8)
    shifted[r:r+n, r:r+m] = binary
    
    res = np.zeros((n, m), dtype = np.bool)

    for i in range(r, r+n):
        for j in range(r, r+m):
            if (_is_intersection(shifted, i, j)):
                res[(i - r, j - r)] = 1

    return res

def intersectionDetection(img_binary):
    """
        输入二值图，输出一个mask，表示哪些点位于交叉附近。
        Args:
            img_binary (ndarray) : 单通道二值图
        Returns:
            mask (ndarray) 
    """
    return find_intersection(img_binary, r = 20)

if (__name__ == "__main__"):
    
    img = cv2.imread('binary.jpg', cv2.IMREAD_GRAYSCALE)
    binary = (img > 127).astype(np.uint8)
    
    ans = intersectionDetection(binary)

    binary[ans] = 127
    plt.imshow(binary, cmap = plt.cm.gray)
    #plt.plot(ans[:, 1], ans[:, 0], markersize = 1, c = 'red', marker = '.', linewidth = 0)
    plt.savefig("interesection.pdf", dpi = 1200)
    
    #savemat("end_points.mat", {"end_points" : ans})





