import numpy as np
import cv2
from math import *
from lib.hair import get_width,getWarpTile
from lib.utils import midUpsample,findNearest


class BasicProcess():

    def magnet(self,point,img_binary,size = 16):
        """[吸铁石，点坐标修正]

        Args:
            point (list):   [点坐标]      样例 --> [x,y]
            img_binary (ndarray): [整图二值图]
            size (int, optional): [搜索尺寸]. Defaults to 32.

        Returns:
            point (list):   [点坐标]      样例 --> [x,y]
        """        
        return point

    def waist(self,joints,img_binary):
        """[自动检测宽度]

        Args:
            joints (list):   [骨架点序列]      样例 --> [[x,y],[x,y],[x,y]]
            img_binary (ndarray): [整图二值图]

        Returns:
            (list): [骨架点序列]
        """        
        joints = midUpsample(joints)
        waist_array = np.zeros(len(joints) - 1)
        for i in range(len(joints) - 1):
            x1 = joints[i][0]
            y1 = joints[i][1]
            x2 = joints[i + 1][0]
            y2 = joints[i + 1][1]
            if x2 - x1 == 0:
                angle = 90
            elif y2 - y1 == 0:
                angle = 0
            else:
                angle = atan(-(y2 - y1) / (x2 - x1)) * 57.3
                if angle < 0:
                    angle += 180
            mid_point = ((x1 + x2) / 2, (y1 + y2) / 2)
            result = getWarpTile(img_binary, mid_point, 20, 5,
                                 rect_angle=angle / 57.3, inverse=True)
            tile = result['tile']
            # 测宽
            [waist, y_offset] = get_width(tile)
            waist_array[i] = waist
        waist = round(findNearest(waist_array, np.median(waist_array)))
        return waist

    def border(self,joints,img_binary):
        """[修正骨架点]

        Args:
            joints (list):   [骨架点序列]      样例 --> [[x,y],[x,y],[x,y]]
            img_binary (ndarray): [整图二值图]

        Returns:
            (list):   [骨架点序列] 
        """        
        return joints



class MyProcess(BasicProcess):
    # 吸铁石
    def magnet(self,point, img_binary, size=16):
        """
        point: the loc of the raw point
        img_binary: the binary img
        return point2: the calculated point
        """
        # 确定直线矩阵
        point = point[::-1]
        # print(point)
        width,length = img_binary.shape

        # print('bi大小:\n',width,length)
        m1 = point[0] - size
        if m1 < 0:
            m1 = 0
        m2 = point[0] + size + 1
        if m2 > width:
            m2 = width
        m3 = point[1] - size
        if m3 <0:
            m3=0
        m4 = point[1] + size + 1
        if m4 > length:
            m4 = length
        #以上防止范围超出图片大小
        # img_binary = img_binary[::-1]
        line_cloumn = img_binary[m1:m2, point[1]]
        line_row = img_binary[point[0], m3:m4]

        # 确定垂直线第一个和最后一个数值最高的点的坐标
        indx = cv2.minMaxLoc(line_cloumn, None)
        indx2 = np.where(line_cloumn == indx[1])
        l0 = indx2[0]  # 所有最大值点的坐标
        l = np.size(indx2, 1)  # 最小值的个数
        firstmin_loc = indx[3]
        firstmin_loc = np.array([firstmin_loc[1], 0])
        lastmin_loc = np.array([l0[l - 1], 0])
        # 确定中间点的坐标
        mid = (firstmin_loc + lastmin_loc) / 2
        mid = np.around(mid)
        if mid[0] == size:
            mid = firstmin_loc
        row = mid[0] - size
        point2_column = np.array([point[0] + row, point[1]])
        # print('垂点：\n',point2_column)

        # 确定水平线第一个和最后一个数值最大的点的坐标
        indx3 = cv2.minMaxLoc(line_row, None)
        indx4 = np.where(line_row == indx3[1])
        l01 = indx4[0]  # 所有最大值点的坐标
        l11 = np.size(indx4, 1)  # 最小值的个数
        firstmin_loc1 = indx3[3]
        firstmin_loc1 = np.array([0, firstmin_loc1[1]])
        lastmin_loc1 = np.array([0, l01[l11 - 1]])
        # 确定中间点的坐标
        mid1 = (firstmin_loc1 + lastmin_loc1) / 2
        mid1 = np.around(mid1)
        if mid1[1] == size:
            mid1 = firstmin_loc1
        row1 = mid1[1] - size
        point2_column1 = np.array([point[0], point[1] + row1])

        # print('水平点：\n',point2_column1)
        if abs(row) <= abs(row1):
            point2 = point2_column
        else:
            point2 = point2_column1

        point2 = point2[::-1]
        # print(point2)
        return point2

