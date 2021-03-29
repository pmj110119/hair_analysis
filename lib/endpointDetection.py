#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
from matplotlib import pyplot as plt
#from scipy.io import savemat


def erosion(src, erosion_size = 3):
    erosion_shape = cv2.MORPH_RECT
    element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                        (erosion_size, erosion_size))
    return cv2.erode(src, element)
    #cv.imshow(title_erosion_window, erosion_dst)

class UFS: # a union-find set（并查集）
    def __init__(self):
        self.fa = {}
    def getfa(self, i):
        if (isinstance(self.fa[i], int)):
            return i
        res = self.getfa(self.fa[i])
        self.fa[i] = res
        return res
    def create_node(self, i):
        if (i not in self.fa):
            self.fa[i] = -1
    def merge(self, i, j):
        self.create_node(i)
        self.create_node(j)
        i, j = self.getfa(i), self.getfa(j)
        if (i == j):
            return
        if (self.fa[i] < self.fa[j]):
            self.fa[i] += self.fa[j]
            self.fa[j] = i
        else:
            self.fa[j] += self.fa[i]
            self.fa[i] = j
    
def _merge_endpoint(binary, candidate, r):
    '''
    Remove redundant hits for a single endpoint in the binary image.
    '''
    n, m = binary.shape
    
    q = np.zeros( (4 * r * r, 2), dtype = np.int32 )
    vis = np.zeros((n, m), dtype = bool)
    ufs = UFS()
    
    def _bfs(x0, y0):
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
                        if ((xx ,yy) in candidate):
                            ufs.merge((xx, yy), (x0, y0))
                        q[tail] = (xx, yy)
                        tail += 1
                        
        for i in range(tail):
            vis[q[i][0], q[i][1]] = False
                        
    for x, y in candidate:
        ufs.create_node((x, y))
        _bfs(x, y)
    
    ans = {}
    for x, y in candidate:
        xx, yy = ufs.getfa((x, y))
        if ((xx, yy) not in ans):
            ans[(xx, yy)] = np.array([0, 0], dtype = np.int32)
        #if (candidate[x, y] < candidate[xx, yy]):
        ans[(xx, yy)] += (x, y)
    
    for x, y in candidate:
        if (isinstance(ufs.fa[(x, y)], int)):
            res = (ans[(x, y)] / -ufs.fa[(x, y)])
            ans[(x, y)] = (round(res[0]), round(res[1]))
    
    return np.array(list(ans.values()))

def find_endpoint(binary, r = 10, degree_thres = 60):
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
    
    n, m = binary.shape
    
    tmp = np.arange(1, r + 1)[np.newaxis, :]
    degree = (np.arange(0, 360)/360. * 2 * np.pi)[:, np.newaxis]
    dx = np.floor(tmp * np.cos(degree)).astype(np.int32)
    dx_ = np.ceil(tmp * np.cos(degree)).astype(np.int32)
    dy = np.floor(tmp * np.sin(degree)).astype(np.int32)
    dy_ = np.ceil(tmp * np.sin(degree)).astype(np.int32)
    del tmp
    
    def _is_endpoint(binary, x, y):
        
        if (binary[x, y] == 0) or ((binary[x-1, y] & binary[x, y-1] & binary[x+1, y] & binary[x, y+1]) == 1):
            return None

        values = np.max( (binary[x + dx, y + dy], binary[x + dx_, y + dy], 
                          binary[x + dx, y + dy_], binary[x + dx_, y+dy_]), axis = 0 )
        flag = values.sum(axis = 1) == r
        s = (flag[1:] != flag[:-1]).sum() + (flag[0] != flag[-1])
        angle = flag.sum()

        if (s == 2) and (angle < degree_thres):
            return angle
        else:
            return None

    shifted = np.zeros((n + 2 * r + 1, m + 2 * r + 1), dtype = np.uint8)
    shifted[r:r+n, r:r+m] = binary
    candidates = {}
    for i in range(r, r+n):
        for j in range(r, r+m):
            angle = _is_endpoint(shifted, i, j)
            if (angle is not None):
                candidates[(i - r, j - r)] = angle
    
    return _merge_endpoint(binary, candidates, 10)
    #return candidates

def endpointDetection(img_binary):
    """
        输入二值图，输出所有检测到的端点。
        Args:
            img_binary (ndarray) : 单通道二值图
        Returns:
            points (ndarray) : 所有检测到的端点
    """
    return find_endpoint(img_binary, r = 15, degree_thres = 60)
    #candidates = find_endpoint(img_binary, r = 15, degree_thres = 60)
    #candidates_ = find_endpoint(erosion(img_binary, 3), r = 15, degree_thres = 60)
    #candidates.update( candidates_ )
    #return _merge_endpoint(img_binary, candidates, 10)

if (__name__ == "__main__"):
    
    img = cv2.imread('binary.jpg', cv2.IMREAD_GRAYSCALE)
    binary = (img > 127).astype('uint8')
    
    ans = endpointDetection(binary)

    plt.imshow(binary, cmap = plt.cm.gray)
    plt.plot(ans[:, 1], ans[:, 0], markersize = 1, c = 'red', marker = '.', linewidth = 0)
    plt.savefig("endpoints.pdf", dpi = 1200)
    
    #savemat("end_points.mat", {"end_points" : ans})





