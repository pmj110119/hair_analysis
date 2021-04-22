#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import time
from numba import jit
from numba.typed import List, Dict


#from scipy.io import savemat


# def erosion(src, erosion_size = 3):
#     erosion_shape = cv2.MORPH_RECT
#     element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
#                                         (erosion_size, erosion_size))
#     return cv2.erode(src, element)
#     #cv.imshow(title_erosion_window, erosion_dst)

# class UFS: # a union-find set（并查集）
#     def __init__(self):
#         self.fa = {}
#     def getfa(self, i):
#         if (isinstance(self.fa[i], int)):
#             return i
#         res = self.getfa(self.fa[i])
#         self.fa[i] = res
#         return res
#     def create_node(self, i):
#         if (i not in self.fa):
#             self.fa[i] = -1
#     def merge(self, i, j):
#         self.create_node(i)
#         self.create_node(j)
#         i, j = self.getfa(i), self.getfa(j)
#         if (i == j):
#             return
#         if (self.fa[i] < self.fa[j]):
#             self.fa[i] += self.fa[j]
#             self.fa[j] = i
#         else:
#             self.fa[j] += self.fa[i]
#             self.fa[i] = j

@jit(nopython = True)
def UFS_getfa(fax, fay, x, y):
    if (fax[x, y] < 0):
        return (x, y)
    else:
        ansx, ansy = UFS_getfa(fax, fay, fax[x, y], fay[x, y])
        fax[x, y] = ansx
        fay[x ,y] = ansy
        return (ansx, ansy)
    
@jit(nopython = True)
def UFS_merge(fax, fay, x1, y1, x2, y2):
    x1, y1 = UFS_getfa(fax, fay, x1, y1)
    x2, y2 = UFS_getfa(fax, fay, x2, y2)
    if (x1 == x2) and (y1 == y2):
        return
    if (fax[x1, y1] < fax[x2, y2]):
        x1, y1, x2, y2 = x2, y2, x1, y1
    fax[x2, y2] += fax[x1, y1]
    fax[x1, y1], fay[x1, y1] = x2, y2

@jit(nopython = True)
def _bfs(binary, is_candidate, vis, q, r, x0, y0, fax, fay):
    
    n, m = binary.shape

    head, tail = 0, 1
    q[0] = (x0, y0)
    vis[x0, y0] = True
    
    while (head < tail):
        x, y = q[head]
        head += 1
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx != 0) or (dy != 0):
                    xx, yy = x + dx, y + dy
                    if (xx < 0) or (xx >= n)  or (yy < 0) or (yy >= m):
                        continue
                    if (binary[xx, yy] == 0) or (vis[xx, yy]) or (abs(xx - x0) >= r) or (abs(yy - y0) >= r):
                        continue
                    vis[xx, yy] = True
                    if (is_candidate[xx, yy]):
                        UFS_merge(fax, fay, xx, yy, x0, y0)
                    q[tail] = (xx, yy)
                    tail += 1
                    
    for i in range(tail):
        vis[q[i][0], q[i][1]] = False

@jit(nopython = True)
def _merge_endpoint(binary, candidate, r):
    '''
    Remove redundant hits for a single endpoint in the binary image.
    '''
    binary = binary.astype(np.uint8)
    n, m = binary.shape
    
    q = np.zeros( (4 * r * r, 2), dtype = np.int32 )
    vis = np.zeros((n, m), dtype = np.uint8)

    is_candidate = np.zeros((n, m), dtype = np.uint8)
    for x, y in candidate:
        is_candidate[x, y] = True

    fax = np.ones((n, m), dtype = np.int32) * -1
    fay = np.ones((n, m), dtype = np.int32) * -1
    
    for x, y in candidate:
        _bfs(binary, is_candidate, vis, q, np.int32(r), np.int32(x), np.int32(y), fax, fay)
    
    ans = Dict()
    for x, y in candidate:
        xx, yy = UFS_getfa(fax, fay, x, y)
        xx, yy = np.int32(xx), np.int32(yy)
        if ((xx, yy) not in ans):
            ans[(xx, yy)] = np.array([0, 0], dtype = np.int32)
        ans[(xx, yy)] += np.array([x, y], dtype = np.int32)
    
    final_res = List()
    for x, y in candidate:
        if (fax[x, y] < 0):
            res = (ans[(x, y)] / -fax[(x, y)])
            ansx, ansy = np.int32(round(res[0])), np.int32(round(res[1]))
            final_res.append((ansx, ansy))
    return final_res

@jit(nopython = True)
def _check(binary, x, y):
    xx, yy = int(np.floor(x)), int(np.floor(y))
    xx_, yy_ = int(np.ceil(x)), int(np.ceil(y))
    return binary[xx, yy] or binary[xx_, yy] or binary[xx, yy_] or binary[xx_, yy_]

@jit(nopython = True)
def _find_endpoint(binary, r, degree_thres):
    n, m = binary.shape
    
    shifted = np.zeros((n + 2 * r + 1, m + 2 * r + 1), dtype = np.uint8)
    shifted[r:r+n, r:r+m] = binary
    
    is_candidate = np.zeros((n, m), dtype = np.uint8)
    candidate = List()
    for i in range(r, r + n):
        for j in range(r, r + m):
            if (shifted[i, j] == 0)  or ((shifted[i-1, j] & shifted[i, j-1] & shifted[i+1, j] & shifted[i, j+1]) == 1):
                continue
            lastflag = -1
            s = 0
            angle = 0
            for degree in range(361):
                rad = degree / 360 * 2 * np.pi
                flag = 1
                for k in range(1, r+1):
                    if not _check(shifted, i + np.cos(rad) * k, j + np.sin(rad) * k):
                        flag = 0
                        break
                s += (flag != lastflag)
                lastflag = flag
                angle += (flag)
            s -= 1
            angle -= (lastflag)
            if (s == 2) and (angle < degree_thres):
                is_candidate[i - r, j - r] = 1
                candidate.append((np.int32(i-r), np.int32(j-r)))
    return candidate, is_candidate

def find_endpoint(binary, r = 15, degree_thres = 60, endpoints = None):
    '''
    Find all of the end points of the hairs in a given binary image.
    
    Args:
    ------
    binary : 2D numpy.array. A binary image.
    r : int, optional. Should be in [1, 30]. The radius of the neibourhood circle.
    degree_thres : float, optional. 
    
    Returns:
    ------
    endpoints : 2D numpy.array, [(x1, y1), (x2, y2), ...] Each row represents an end point. 
    '''
    
    # n, m = binary.shape
    
    # tmp = np.arange(1, r + 1)[np.newaxis, :]
    # degree = (np.arange(0, 360)/360. * 2 * np.pi)[:, np.newaxis]
    # dx = np.floor(tmp * np.cos(degree)).astype(np.int32)
    # dx_ = np.ceil(tmp * np.cos(degree)).astype(np.int32)
    # dy = np.floor(tmp * np.sin(degree)).astype(np.int32)
    # dy_ = np.ceil(tmp * np.sin(degree)).astype(np.int32)
    # del tmp

    # def _is_endpoint(binary, x, y):
    #     if (binary[x, y] == 0) or ((binary[x-1, y] & binary[x, y-1] & binary[x+1, y] & binary[x, y+1]) == 1):
    #         return False
    #     values = np.max( (binary[x + dx, y + dy], binary[x + dx_, y + dy], 
    #                     binary[x + dx, y + dy_], binary[x + dx_, y + dy_]), axis = 0 )
    #     flag = values.sum(axis = 1) == r
    #     s = (flag[1:] != flag[:-1]).sum() + (flag[0] != flag[-1])
    #     angle = flag.sum()
    #     return (s == 2) and (angle < degree_thres)
    
    # shifted = np.zeros((n + 2 * r + 1, m + 2 * r + 1), dtype = np.uint8)
    # shifted[r:r+n, r:r+m] = binary
    # candidates = set()
    # for i in range(r, r+n):
    #     for j in range(r, r+m):
    #         if (_is_endpoint(shifted, i, j)):
    #             candidates.add( (i - r, j - r) )

    binary = np.array(binary, dtype = np.uint8)
    candidate, is_candidate = _find_endpoint(binary, r, degree_thres)

    if (endpoints is not None):
        for x, y in endpoints:
            if not is_candidate[x, y]:
                candidate.append((np.int32(x), np.int32(y)))
    #print("before merge", time() - t0, "s")
    return np.array(magnet(binary, _merge_endpoint(binary, candidate, 10)), dtype = np.int32)

#@jit(nopython = True)
def magnet(binary, candidate):

    n, m = binary.shape

    final_res = List()
    for ansx, ansy in candidate:
        if (ansx < 0) or (ansx >= n) or (ansy < 0) or (ansy >= m):
            continue
        if (binary[ansx, ansy] == 0):
            tmp, tmpdist = (-1, -1), -1.0
            for dx in range(  max(-10, -ansx), min(+11, n-ansx) ):
                for dy in range( max(-10, -ansy), min(+11, m-ansy) ):
                    ansx_, ansy_ = ansx + dx, ansy + dy
                    curdist = math.sqrt(dx**2 +dy**2)
                    if (binary[ansx_, ansy_] == 1) and ((tmpdist < 0) or ( curdist < tmpdist )):
                        tmp, tmpdist = (ansx_, ansy_), curdist
            if (tmpdist > 0):
                final_res.append( tmp )
        else:
            final_res.append((ansx, ansy))
    
    return final_res

def endpointDetection(img_binary, endpoints = None, refind = True):
    """
        输入二值图，输出所有检测到的端点。
        Args:
            img_binary (ndarray) : 单通道二值图
            endpoints (ndarray, N * 2, optional) : 建议的点
        Returns:
            points (ndarray) : 所有检测到的端点
    """
    if (refind == False):
        return np.array(magnet(img_binary, endpoints), dtype = np.int32)
    return find_endpoint(img_binary, r = 15, degree_thres = 60, endpoints = endpoints)

if (__name__ == "__main__"):
    
    from time import time
    
    img = cv2.imread('binary.jpg', cv2.IMREAD_GRAYSCALE)
    binary = (img > 127).astype(np.uint8)
    binary = binary[:200, 0:200]
    
    t0 = time()
    ans = endpointDetection(binary)
    print(time() - t0, "s")
    t0 = time()
    ans = endpointDetection(binary)
    print(time() - t0, "s")

    plt.imshow(binary, cmap = plt.cm.gray)
    plt.plot(ans[:, 1], ans[:, 0], markersize = 1, c = 'red', marker = '.', linewidth = 0)
    #plt.show()
    plt.savefig("endpoints.pdf", dpi = 1200)
    
    #savemat("end_points.mat", {"end_points" : ans})





