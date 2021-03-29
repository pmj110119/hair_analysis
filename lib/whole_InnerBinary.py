# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:36:28 2021

@author: JY
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
from skimage import morphology
from tqdm import tqdm, trange

def innerBinary(img_bgr, point):
    """
        Args:
            img_bgr(ndarray):[三通道图像]
            point(list):     [点坐标]
        Returns:
            binary(ndarray): [单通道图像]
    """
    
    I = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # print(I)
    I_sizes = I.shape
    # print(I_sizes)
    x = point[0]
    y = point[1]
    J = np.zeros(I_sizes,dtype=np.uint8)
    # print(I[x,y])
    reg_mean = I[x,y]  
    # x = 122
    # y = 132
    # print(x,y)
    # print(point[0],point[1])
    # print(I[point[0],point[1]])
    # print(I[122,132])
    # print(I[x,y])
    reg_size = 1
    neg_free = 10000
    neg_free_add = 10000
    neg_list = np.zeros((neg_free,3),dtype=np.int32)
    neg_pos = 0
    pixdist = 0
    neigb = np.array([[-1,0],[1,0],[0,-1],[0,1]])
    # print(neigb.shape)
    
    #新变量
    pixdist_9 = 0
    reg_maxdist_9 = 0.06*255
    reg_maxdist = 0.06*255
    reg_mean_9 = 0
    J_add = np.zeros((I_sizes[0]+2, I_sizes[1]+2),dtype=np.uint8)
    I_add = np.zeros((I_sizes[0]+2, I_sizes[1]+2),dtype=np.uint8)
    I_add[1:I_sizes[0]+1, 1:I_sizes[1]+1] = I[0:I_sizes[0],0:I_sizes[1]]
    
    # print(x,y)
    while ((pixdist < reg_maxdist or pixdist_9 < reg_maxdist_9) and reg_size < I.size):
        for j in range(4):
            x_n = x + neigb[j,0]
            y_n = y + neigb[j,1]
            ins = (x_n>=0)and(y_n>=0)and(x_n<=I_sizes[0]-1)and(y_n<=I_sizes[1]-1);
            if( ins and J[x_n,y_n]==0):
                neg_pos = neg_pos+1
                neg_list[neg_pos-1,:] =[ x_n, y_n, I[x_n,y_n]]
                J[x_n,y_n] = 1
                
        if (neg_pos+10>neg_free):
            
            neg_list_add = np.zeros((neg_free_add,3),dtype=np.int32)
            neg_list = np.row_stack((neg_list,neg_list_add))
            
        dist = neg_list[0:neg_pos,2]-reg_mean;
        index = np.argmin(dist)
        pixdist = dist[index]
        
        #计算区域的新的均值
        reg_mean = (reg_mean * reg_size +neg_list[index,2])/(reg_size + 1)
        reg_size = reg_size + 1;
        
        #将旧的种子点标记为已经分割好的区域像素点
        J[x,y]=2 #标志该像素点已经是分割好的像素点
        x = neg_list[index,0]
        y = neg_list[index,1]
        
        #求reg_mean_9
        J_add[1:I_sizes[0]+1, 1:I_sizes[1]+1] =  J[0:I_sizes[0],0:I_sizes[1]];
        J_9 = (J_add[x:x+3, y:y+3] == 2);
        I_9 = I_add[x:x+3, y:y+3];
        # print(J_9)
        I_9 = J_9*I_9;
        # print(I_9)
        reg_mean_9 = np.sum(I_9)/(np.sum(I_9 != 0)+np.finfo(float).eps);
        
        #求pixdist_9
        pixdist_9 = neg_list[index, 2] - reg_mean_9
        
        #将新的种子点从待分析的邻域像素列表中移除
        neg_list[index,:] = neg_list[neg_pos-1,:]
        neg_pos = neg_pos -1
        
    binary = np.uint8((J==2)*255)
    # print(binary.dtype)
    return binary


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

    def get_weight_matrix():
        w = np.zeros(shape=(11 * 11 * 11, 3))
        m = 0
        for i in np.linspace(0, 1, 11):
            for j in np.linspace(0, 1, 11):
                for k in np.linspace(0, 1, 11):
                    w[m] = [i, j, k]
                    m += 1
        return w

    def get_gray(im_bgr_part, w):
        z = im_bgr_part.dot(w.T)
        var_arr = np.std(z, axis=(0, 1))
        var_index = np.argmax(var_arr)
        z_gray = z[:, :, var_index]
        z_gray = im2double(z_gray)
        z_gray = np.uint8(z_gray * 255)
        return z_gray

    def remove_noise(img):
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        quantile = np.percentile(stats[:, 4], 92)
        q = (stats[:, 4] < quantile)
        w_h = stats[:, 2] / stats[:, 3] > 1.5
        h_w = stats[:, 3] / stats[:, 2] > 1.5
        sel = q
        # sel = q & (~(w_h | h_w))
        for x, y, w, h, s in stats[sel]:
            label = labels[y:y + h, x:x + w]
            lab = label.reshape(-1, )
            lab = np.unique(lab)
            lab = np.setdiff1d(lab, 0)
            for l in lab:
                seeds = np.argwhere(label == l)
                seedlist = list(seeds)
                # print(seedlist)
                if len(seedlist) == s:
                    for point in seedlist:
                        img[y:y + h, x:x + w][point[0], point[1]] = 0
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
            im_bgr_part_gray = cv2.cvtColor(im_bgr_part, cv2.COLOR_BGR2GRAY)
            par = np.arange(12, 13, 0.01)
            per = np.zeros(shape=(np.arange(12, 13, 0.01).size,))
            m = 0
            total = np.arange(12, 13, 0.01).size
            for minus in np.arange(12, 13, 0.01):
                binary = cv2.adaptiveThreshold(im_bgr_part_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, minus)
                per[m] = np.sum(binary == img_mask_part) / size
                if np.sum(binary == img_mask_part) / size > 0.98:
                    #print(par[m], per[m], m)
                    break
                elif m == (total - 1):
                    max_index = np.argmax(per)
                    minus_val = par[max_index]
                    binary = cv2.adaptiveThreshold(im_bgr_part_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, minus_val)
                    #print(par[max_index], per[max_index], m)
                m = m + 1
            out_pic[i:i + h_size, j:j + w_size] = binary
            
    out_pic = remove_noise(out_pic)
    
    # print(out_pic.dtype)
    # print(binary.dtype)
    return out_pic

#@nb.jit()
def whole_InnerBinary(img_bgr, img_mask):
    """
    :param img_bgr: 读入的原图像
    :param img_mask: 深度学习生成的二值图
    :return: 内点生长全图         
    """
    
    #得到较为优质的二值图作为寻找内点的参考
    bw = local_adaptive(img_bgr, img_mask)
    
    # #对bw进行闭运算    
    kernel = np.ones((2, 2), np.uint8)
    # for i in [0,2]:
    #     for j in [0,2]:
    #         kernel[i,j] = 0
    # print(kernel)
    # bw_close = cv2.morphologyEx(bw，cv2.MORPH_CLOSE, kernel)
    
    #腐蚀
    bw = cv2.erode(bw, kernel)
    
    #提取骨架
    # bw[bw==255] = 1
    # bw = morphology.skeletonize(bw)
    # bw = bw.astype(np.uint8)*255

    
    #创建结果大图
    binary = np.zeros(bw.shape,dtype = np.uint8)
    
    #调试使用
    m = 0
    
    #划分小图
    row, col = bw.shape[0:2]
    row_s = 600
    col_s = 600
    row_size = row_s
    col_size = 0
    for i in range(0, row, row_s):
        if i == np.arange(0, row, row_s)[-1]:
            row_size = row - i
        for j in np.arange(0, col, col_s):
            if j == np.arange(0, col, col_s)[-1]:
                col_size = col - j
            else:
                col_size = col_s
            bw_part = bw[i:i+row_size,j:j+col_size]
            img_bgr_part = img_bgr[i:i+row_size,j:j+col_size]
            binary_part = np.zeros(bw_part.shape)
            ind = np.argwhere(bw_part == 255)
            for point in ind:
                if binary_part[point[0],point[1]] == 1:
                    continue
                binary_part_1 = innerBinary(img_bgr_part, point)
                binary_part = np.logical_or(binary_part, binary_part_1)
                m = m+1;
            binary_part = np.uint8(binary_part)*255
            binary[i:i+row_size,j:j+col_size] = binary_part

    # #调试用
    # print("生长次数：",m)
    
    
    # #直接全图
    # ind = np.argwhere(bw == 255)
    # for index,point in tqdm(enumerate(ind),total=len(ind)):
    #     if binary[point[0],point[1]] == True:
    #         continue
    #     binary_one_point = innerBinary(img_bgr, point)
    #     binary = np.logical_or(binary, binary_one_point)
    #     m = m+1
    binary = np.uint8(binary)*255
    # print("生长次数：",m)
    return binary
    # return bw
       
    
if __name__ == '__main__':
    img_bgr = cv2.imread("124DDBSY.jpg")
    img_mask = cv2.imread("124DDBSY.jpg.png", cv2.IMREAD_GRAYSCALE)
    binary = whole_InnerBinary(img_bgr, img_mask)
    cv2.imwrite("124DDBSY.jpg.inner.png", binary)
    plt.imshow(binary, cmap='Greys_r')
    plt.axis('off')
    plt.title(binary)
    plt.show()
    
    
    
    