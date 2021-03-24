# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 19:45:22 2021

@author: JY
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def innerBinary(img_bgr, point, debug=False):
    """
        Args:
            img_bgr(ndarray):[三通道图像]
            point(list):     [点坐标]
        Returns:
            binary(ndarray): [单通道图像]
    """
    # if debug:
    #     temp = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR).copy()
    #     cv2.circle(temp,(point[0],point[1]),2,(0,255,0),2)
    #     cv2.imshow('innerDebug',temp)
    #     cv2.waitKey(1)



    I = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # print(I)
    I_sizes = I.shape
    # print(I_sizes)
    x = point[1]
    y = point[0]
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
    reg_maxdist = 0.1*255
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


def On_LBUTTONDOWN(event, x, y, flags, param,):
    if event == cv2.EVENT_LBUTTONDOWN:
       
        point[0],point[1] = y,x 
        print(point)
        xy = "%d,%d" % (x, y)
        cv2.circle(img_bgr, (x, y), 1, (255, 0, 0), thickness = -1)
        cv2.putText(img_bgr, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0,0,0), thickness = 1)
        cv2.imshow("image", img_bgr)
        
        binary = innerBinary(img_bgr_1, point)
        plt.imshow(binary,cmap='Greys_r')
        plt.title('binary') 
        plt.axis('off')
        plt.show()
        # print(2)
        # print(x,y)

if __name__ == "__main__":
    point = np.zeros(2,dtype = np.int32)
    # point = np.zeros(2,dtype = np.int)
    # point = np.array([0,0],dtype=np.int)
    img_bgr = cv2.imread("pic_2.jpg")
    img_bgr_1 = img_bgr.copy()
    
    # cv2.namedWindow("image")
    # cv2.setMouseCallback("image", On_LBUTTONDOWN)
    # cv2.imshow("image",img_bgr)
    # cv2.waitKey(0)
    # print(point)
    # cv2.destroyAllWindows()
    # binary = innerBinary(img_bgr_1, point)
    # plt.imshow(binary,cmap='Greys_r')
    # plt.title('binary') 
    # plt.axis('off')
    # plt.show()
    # cv2.destroyAllWindows()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", On_LBUTTONDOWN)
    # cv2.imshow("image",img_bgr)
    while(1):
        # print(1)
        cv2.imshow('image',img_bgr)
        if cv2.waitKey(0)&0xFF==27:
            break
    cv2.destroyAllWindows()
    
    # cv2.imshow("image", binary)

    # cv2.waitKey(0)
    
    