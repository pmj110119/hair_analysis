# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:36:28 2021

@author: JY

Updated by CDC.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
from numba import jit
#from numba.typed import List

#rom skimage import morphology

def local_adaptive(img_bgr, img_mask):
    """
    :param img_bgr: 读入的原图像
    :param img_mask: 深度学习生成的二值图
    :return: 局部二值化得到的图像
    """
    def im2double(im):
        if im.dtype == 'float':
            out = cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # 归一化
        else:
            out = im.astype('float') / 255
        return out

    # def get_weight_matrix():
    #     w = np.zeros(shape=(11 * 11 * 11, 3))
    #     m = 0
    #     for i in np.linspace(0, 1, 11):
    #         for j in np.linspace(0, 1, 11):
    #             for k in np.linspace(0, 1, 11):
    #                 w[m] = [i, j, k]
    #                 m += 1
    #     return w

    def get_gray(im_bgr_part, w):
        z = im_bgr_part.dot(w.T)
        var_arr = np.std(z, axis=(0, 1))
        var_index = np.argmax(var_arr)
        z_gray = z[:, :, var_index]
        z_gray = im2double(z_gray)
        z_gray = np.uint8(z_gray * 255)
        return z_gray

    def remove_noise(img):
        # Optimized by CDC
        n, m = img.shape
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
        # quantile = np.percentile(stats[:, 4], 95)
        # print(quantile)
        # q = (stats[:, 4] < quantile)
        # w_h = stats[:, 2] / stats[:, 3] > 1.5
        # h_w = stats[:, 3] / stats[:, 2] > 1.5
        # sel = q
        # sel = q & (~(w_h | h_w))
        # sel = (stats[:, 4] < 200)
        # for x, y, w, h, s in stats[sel]:
        #     label = labels[y:y + h, x:x + w]
        #     lab = label.reshape(-1, )
        #     lab = np.uniheap(lab)
        #     lab = np.setdiff1d(lab, 0)
        #     for l in lab:
        #         seeds = np.argwhere(label == l)
        #         seedlist = list(seeds)
        #         # print(seedlist)
        #         if len(seedlist) == s:
        #             for point in seedlist:
        #                 img[y:y + h, x:x + w][point[0], point[1]] = 0
        for i in range(1, num_labels):
            y_min, x_min, width, height, area = stats[i]
            if (area < 200):
                x_max = min(x_min + height, n)
                y_max = min(y_min + width, m)
                img[x_min:x_max, y_min:y_max][ labels[x_min:x_max, y_min:y_max] == i ] = 0

        return img


    # img_bgr = im2double(img_bgr)
    img_mask[img_mask < 10] = 0
    img_mask[img_mask > 200] = 255
    # w = get_weight_matrix()
    height, width = img_bgr.shape[0:2]
    if height < 400 and width < 400:
        h_s = height // 2
        w_s = width // 2
    elif height < 400:
        h_s = height // 2
        w_s = 400
    elif width < 400:
        h_s = 400
        w_s = width // 2
    else:
        h_s = 400
        w_s = 400
    h_size = h_s
    out_pic = np.zeros(shape=[height, width], dtype=np.uint8)
    for i in np.arange(0, height, h_s):
        if i == np.arange(0, height, h_s)[-1]:
            h_size = height - i
        for j in np.arange(0, width, w_s):
            if j == np.arange(0, width, w_s)[-1]:
                w_size = width - j
            else:
                w_size = w_s
            im_bgr_part = img_bgr[i:i + h_size, j:j + w_size]
            img_mask_part = img_mask[i:i + h_size, j:j + w_size]
            size = img_mask_part.size
            # im_bgr_part_gray = get_gray(im_bgr_part, w)
            ## im_bgr_part_gray = cv2.cvtColor(im_bgr_part, cv2.COLOR_BGR2GRAY)
            im_bgr_part_gray = bgr2grey(im_bgr_part)

            # ---- updated by CDC. -------
            #par = np.arange(12, 13, 0.01)
            #per = np.zeros(shape=(np.arange(12, 13, 0.01).size,))
            #m = 0
            #total = np.arange(12, 13, 0.01).size
            maxPercentage, maxPercentage_binary = 0., None
            for minus in np.arange(12, 13, 0.01):
                binary = cv2.adaptiveThreshold(im_bgr_part_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, minus)
                #per[m] = np.sum(binary == img_mask_part) / size
                percentage = np.sum(binary == img_mask_part) / size
                if percentage > maxPercentage:
                    maxPercentage = percentage
                    maxPercentage_binary = binary
                if percentage > 0.98:
                    break
                # elif m == (total - 1):
                #     max_index = np.argmax(per)
                #     minus_val = par[max_index]
                #     binary = cv2.adaptiveThreshold(im_bgr_part_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, minus_val)
                #     print(par[max_index], per[max_index], m)
                # m = m + 1

            out_pic[i:i + h_size, j:j + w_size] = maxPercentage_binary
            
    out_pic = remove_noise(out_pic)
    
    # print(out_pic.dtype)
    # print(binary.dtype)
    return out_pic

# Modified by CDC. (2021.6.22)
def innerBinary(img_bgr, point):
    """
        Args:
            img_bgr(ndarray):[三通道图像]
            point(list):     [点坐标]
        Returns:
            binary(ndarray): [单通道图像]
    """
    
    #I = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    I = bgr2grey(img_bgr)
    n, m = I.shape
    x, y = point
    #points_in_hair = BFS(I, x, y, 0.02*255, 0.02*255)
    binary = np.zeros((n, m), dtype = np.uint8)
    #binary[points_in_hair[:, 0], points_in_hair[:, 1]] = 255
    BFS(I, x, y, 0.02*255, 0.02*255, binary)
    return binary * 255

    # # print(I)
    # I_sizes = I.shape
    # # print(I_sizes)
    # x, y = point
    # J = np.zeros(I_sizes,dtype=np.uint8)
    # # print(I[x,y])
    # reg_mean = I[x,y]  
    # # x = 122
    # # y = 132
    # # print(x,y)
    # # print(point[0],point[1])
    # # print(I[point[0],point[1]])
    # # print(I[122,132])
    # # print(I[x,y])
    # reg_size = 1
    # neg_free = 10000
    # neg_free_add = 10000
    # neg_list = np.zeros((neg_free,3),dtype=np.int32)
    # neg_pos = 0
    # pixdist = 0
    # neigb = np.array([[-1,0],[1,0],[0,-1],[0,1]])
    # # print(neigb.shape)
    
    # #新变量
    # pixdist_9 = 0
    # reg_maxdist_9 = 0.02*255
    # reg_maxdist = 0.02*255
    # #print('局部阈值',reg_maxdist_9)
    # #print('全局阈值',reg_maxdist)
    # reg_mean_9 = 0
    # J_add = np.zeros((I_sizes[0]+2, I_sizes[1]+2),dtype=np.uint8)
    # I_add = np.zeros((I_sizes[0]+2, I_sizes[1]+2),dtype=np.uint8)
    # I_add[1:I_sizes[0]+1, 1:I_sizes[1]+1] = I[0:I_sizes[0],0:I_sizes[1]]
    
    # # print(x,y)
    # while ((pixdist < reg_maxdist or pixdist_9 < reg_maxdist_9) and reg_size < I.size):
    #     for j in range(4):
    #         x_n = x + neigb[j,0]
    #         y_n = y + neigb[j,1]
    #         ins = (x_n>=0)and(y_n>=0)and(x_n<=I_sizes[0]-1)and(y_n<=I_sizes[1]-1)
    #         if( ins and J[x_n,y_n]==0):
    #             neg_pos = neg_pos+1
    #             neg_list[neg_pos-1,:] =[ x_n, y_n, I[x_n,y_n]]
    #             J[x_n,y_n] = 1
                
    #     if (neg_pos+10>neg_free):
            
    #         neg_list_add = np.zeros((neg_free_add,3),dtype=np.int32)
    #         neg_list = np.row_stack((neg_list,neg_list_add))
            
    #     dist = neg_list[0:neg_pos,2]-reg_mean
    #     index = np.argmin(dist)
    #     pixdist = dist[index]
        
    #     #计算区域的新的均值
    #     reg_mean = (reg_mean * reg_size +neg_list[index,2])/(reg_size + 1)
    #     reg_size = reg_size + 1
        
    #     #将旧的种子点标记为已经分割好的区域像素点
    #     J[x,y]=2 #标志该像素点已经是分割好的像素点
    #     x = neg_list[index,0]
    #     y = neg_list[index,1]
        
    #     #求reg_mean_9
    #     J_add[1:I_sizes[0]+1, 1:I_sizes[1]+1] =  J[0:I_sizes[0],0:I_sizes[1]]
    #     J_9 = (J_add[x:x+3, y:y+3] == 2)
    #     I_9 = I_add[x:x+3, y:y+3]
    #     # print(J_9)
    #     I_9 = J_9*I_9
    #     # print(I_9)
    #     reg_mean_9 = np.sum(I_9)/(np.sum(I_9 != 0)+np.finfo(float).eps)
        
    #     #求pixdist_9
    #     pixdist_9 = neg_list[index, 2] - reg_mean_9
        
    #     #将新的种子点从待分析的邻域像素列表中移除
    #     neg_list[index,:] = neg_list[neg_pos-1,:]
    #     neg_pos = neg_pos -1
        
    # binary = np.uint8((J==2)*255)
    # # print(binary.dtype)
    # return binary

# Modified by CDC. (2021.6.21)
def whole_InnerBinary(img_bgr, img_mask):
    """
    :param img_bgr: 读入的原图像
    :param img_mask: 深度学习生成的二值图
    :return: 内点生长全图         
    """
    
    #得到较为优质的二值图作为寻找内点的参考
    bw = local_adaptive(img_bgr, img_mask)
    n, m = bw.shape
    # return bw
    
    # #对bw进行闭运算    
    kernel = np.ones((2, 2), np.uint8)
    # for i in [0,2]:
    #     for j in [0,2]:
    #         kernel[i,j] = 0
    # print(kernel)
    # bw_close = cv2.morphologyEx(bw，cv2.MORPH_CLOSE, kernel)
    
    #腐蚀
    bw = cv2.erode(bw, kernel)

    #img_grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    #binary = BFS(img_grey, 6, 121, 0.02 * 255, 0.02 * 255)
    
    #提取骨架
    # bw[bw==255] = 1
    # bw = morphology.skeletonize(bw)
    # bw = bw.astype(np.uint8)*255

    
    # #创建结果大图
    #img_grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_grey = bgr2grey(img_bgr)
    
    # #调试使用
    # m = 0
    inner_points = np.argwhere(bw == 255).tolist()
    inner_points.sort(key = lambda pos : img_grey[pos[0], pos[1]])
    inner_points = np.array(inner_points, dtype = np.int32)
    return _whole_InnerBinary(n, m, inner_points, img_grey) * 255

@jit('void(uint8[:, :], int32, int32, float64, float64, uint8[:, :])', nopython = True)
def BFS(I, stx, sty, thres_local, thres_global, binary):

    n, m = I.shape

    #vis = set()
    #vis.add((stx, sty))
    #binary = np.zeros((n, m), dtype = np.uint8)
    binary[stx, sty] = 1
    sum_intensity = I[stx, sty] + 0.
    total = 1

    heap = [(I[stx, sty], stx, sty)]
    if (stx + 1 < n):
        heapq.heappush(heap, (I[stx+1, sty], np.int32(stx+1), sty))
    if (sty + 1 < m):
        heapq.heappush(heap, (I[stx, sty+1], stx, np.int32(sty+1)))
    if (stx > 0):
        heapq.heappush(heap, (I[stx-1, sty], np.int32(stx-1), sty))
    if (sty > 0):
        heapq.heappush(heap, (I[stx, sty-1], stx, np.int32(sty-1)))
    heapq.heapify(heap)

    while len(heap):
        _, x, y = heapq.heappop(heap)
        #if ((x, y) in vis):
        if binary[x, y]:
            continue

        if (I[x, y] - sum_intensity / total < thres_global): # 当前点灰度与整个生长区域平均灰度接近
            pass
        else:
            neighbor_intensity_sum = 0.
            neighbor_total = 0
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx = x + dx
                    ny = y + dy
                    if (nx < 0) or (nx >= n) or (ny < 0) or (ny >= m) or (binary[nx, ny] == 0): #((nx, ny) not in vis):
                        continue
                    neighbor_total += 1
                    neighbor_intensity_sum += I[nx, ny]
            
            if (I[x, y] - neighbor_intensity_sum / (neighbor_total + 1e-10) < thres_local):
                # 当前点灰度与其8邻域中已生长点的平均灰度接近 
                pass
            else:
                break
        
        sum_intensity += I[x, y]
        total += 1
        binary[x, y] = 1
        #vis.add((x, y))

        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx = np.int32(x + dx)
            ny = np.int32(y + dy)
            if (nx < 0) or (nx >= n) or (ny < 0) or (ny >= m) or (binary[nx, ny]): #((nx, ny) in vis):
                continue
            heapq.heappush(heap, (I[nx, ny], nx, ny))

    #return np.array(list(vis), dtype = np.int32)

    
@jit('uint8[:, :](int32, int32, int32[:, :], uint8[:, :])', nopython = True)
def _whole_InnerBinary(n, m, inner_points, img_grey):

    binary = np.zeros((n, m), dtype = np.uint8)
    for x, y in inner_points:
        if binary[x, y]:
            continue
        BFS(img_grey, x, y, 0.02*255, 0.02*255, binary)

    #划分小图
    # binary = np.zeros(bw.shape, dtype = np.uint8)
    # row, col = bw.shape[0:2]
    # for x1 in range(0, row, 100):
    #     for y1 in range(0, col, 100):
    #         for x in range(x1, min(row, x1 + 100)):
    #             for y in range(y1, min(col, y1 + 100)):
    #                 if (bw[x, y] == 0) or binary[x, y]:
    #                     continue
    #                 points_in_hair = BFS(img_grey, x, y, 0.02*255, 0.02*255)
    #                 for i in range(points_in_hair.shape[0]):
    #                     binary[points_in_hair[i, 0], points_in_hair[i, 1]] = 255
            # bw_part = bw[i:i+row_size,j:j+col_size]
            # img_bgr_part = img_bgr[i:i+row_size,j:j+col_size]
            # binary_part = np.zeros(bw_part.shape)
            # ind = np.argwhere(bw_part == 255)
            # for point in ind:
            #     if binary_part[point[0],point[1]] == 1:
            #         continue
            #     binary_part_1 = innerBinary(img_bgr_part, point)
            #     binary_part = np.logical_or(binary_part, binary_part_1)
            #     m = m+1
            # binary_part = np.uint8(binary_part)*255
            # binary[i:i+row_size,j:j+col_size] = binary_part   
    
    # #调试用
    # print("生长次数：",m)

    #直接全图
    # ind = np.argwhere(bw == 255)
    # for x, y in ind:
    #     if binary[x, y]:
    #         continue
    #     points_in_hair = BFS(img_grey, x, y, 0.02*255, 0.02*255)
    #     binary[points_in_hair[:, 0], points_in_hair[:, 1]] = 255
 
    #print("生长次数：",m)
    
    return binary

def bgr2grey(I):
    
    return cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    
    # n, m, c = I.shape
    # I = I.reshape(n*m, c)

    # #normalize
    # I = (I - I.mean(axis = 0, keepdims = True)) / I.std(axis = 0, keepdims = True)

    # cov = I.T @ I
    # _, eigV = np.linalg.eigh(cov)
    # I = (I @ eigV[2, :]).reshape(n, m)

    # #convert back to [0, 255]
    # Imin = I.min(axis = 0, keepdims = True)
    # I = (I - Imin) / (I.max(axis = 0, keepdims = True) - Imin) * 255 
    # del Imin
    # return I.astype(np.uint8)

if __name__ == '__main__':

    import time

    img_bgr = cv2.imread("101AAESY.jpg")
    img_mask = cv2.imread("101AAESY.jpg.png", cv2.IMREAD_GRAYSCALE)
    #img_bgr = img_bgr[:400, :400]
    #img_mask = img_mask[:400, :400]

    t0 = time.time()

    #binary = bgr2grey(img_bgr)
    #binary = local_adaptive(img_bgr, img_mask)
    binary = whole_InnerBinary(img_bgr, img_mask)
    #point = (12, 129)
    #point = (231, 203)
    #binary = innerBinary(img_bgr, point)

    print("Time used: ", time.time() - t0, " s")

    cv2.imwrite("103ACDLJ_inner.jpg", binary)
    plt.imshow(binary, cmap='Greys_r')
    plt.axis('off')
    plt.title("binary")
    plt.show()
    
    
    
    