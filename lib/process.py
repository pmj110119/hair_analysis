import numpy as np
import cv2
from math import *
from lib.hair import get_width,getWarpTile
from lib.utils import midUpsample,findNearest
from lib.shortestPath import getShortestPath
from lib.clusterBinary import myAggCluster
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
        waist_median = round(findNearest(waist_array, np.median(waist_array)))
        waist_mean = round(findNearest(waist_array, np.mean(waist_array)))
        waist = min(waist_median,waist_mean)
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

    def binary_search(self,img_bgr, img_binary):
        """[自动调参并筛选后的二值化]

        Args:
            img_bgr (ndarray):  [原图]      
            img_binary (ndarray): [语义分割二值图]

        Returns:
            (ndarray):   [二值图] 
        """        
        return img_binary

    def binart_cluster(self,img_bgr):
        n_clusters = 3
        return myAggCluster(img_bgr,n_clusters)

    def generate_path(self,img_binary, startpoint, endpoint):
       # img_binary = (img_binary==0).astype(np.uint8)

        if False:
            img_test = cv2.cvtColor(img_binary.copy()*255, cv2.COLOR_GRAY2BGR)
            cv2.circle(img_test,(startpoint[1],startpoint[0]),3,(0,255,0),0)
            cv2.circle(img_test, (endpoint[1],endpoint[0]), 3, (0, 0, 255),0)
            cv2.imwrite('zzzz22.png', img_test)

        path_joints, length = getShortestPath(img_binary/255,startpoint,endpoint)
        joints=[]
        step = 15
        for i in np.arange(0, len(path_joints), step):
            joints.append([path_joints[i][1],path_joints[i][0]])
        if len(path_joints)%step!=0:
            joints.append([path_joints[-1][1], path_joints[-1][0]])
        return joints, length

class MyProcess(BasicProcess):
    # 吸铁石
    def magnet(self,point, img_binary, size=12):
        """
        point: the loc of the raw point
        img_binary: the binary img
        return point2: the calculated point
        """
        point2 = [0, 0]
        size = 12
        # point =[1209,912]
        point = point[::-1]  # 输入的点的xy需要修改倒置才能使用
        width, length = img_binary.shape

        # 防止范围超出图片大小
        m1 = point[0] - size  # 行的左界
        if m1 < 0:
            m1 = 0
        m2 = point[0] + size + 1  # 行的右界
        if m2 > width:
            m2 = width
        m3 = point[1] - size  # 列的上界
        if m3 < 0:
            m3 = 0
        m4 = point[1] + size + 1  # 列的下界
        if m4 > length:
            m4 = length

        # 确定矩形
        region = img_binary[m1:m2, m3:m4]
        w, w1 = region.shape

        # 确定那个矩形中存在255点，否则返回原点坐标
        flag = 0
        for x in np.nditer(region):
            if x == 255:
                flag = 1
        if flag == 0:
            print("矩形中不存在最大值点,返回原点\n")
            point2[0] = point[1]
            point2[1] = point[0]

        # 确定矩形中输入点的位置
        center = np.zeros((1, 2), dtype=int)
        if w == w1:  # 大多数正常
            center = [size, size]
        elif m1 == 0 and w1 == 2 * size + 1:  # 靠左
            center = [w - size, size]
        elif m2 == width and w1 == 2 * size + 1:  # 靠右
            center = [size, size]
        elif m3 == 0 and w == 2 * size + 1:  # 靠上
            center = [size, w1 - size]
        elif m4 == length and w == 2 * size + 1:  # 靠下
            center = [size, size]
        elif m1 == 0 and m3 == 0:  # 左上角
            center = [w - size, w1 - size]
        elif m2 == width and m3 == 0:  # 右上角
            center = [size, w1 - size]
        elif m1 == 0 and m4 == length:  # 左下角
            center = [w - size, size]
        elif m2 == width and m4 == length:  # 右下角
            center = [size, size]
        if flag == 1:
            # print("flag=1,region中存在最大值点，输出点为距离最近点\n")
            # 找到矩形中所有最大值的点坐标,距离最小作为输出点
            indx = np.where(region == 255)
            loc1 = indx[0]  # 所有最大值点的行坐标
            loc2 = indx[1]  # 所有最大值点的列坐标
            l3 = loc1 - center[0]
            l4 = loc2 - center[1]
            l3 = np.power(l3, 2)
            l4 = np.power(l4, 2)
            dis = np.sum([l3, l4], axis=0)
            p1 = np.where(dis == np.min(dis))
            p2 = p1[0]
            p3 = p2[0]
            # print("p3:\n", p3)
            point2[0] = point[0] + loc1[p3] - size
            point2[1] = point[1] + loc2[p3] - size
            point2 = point2[::-1]

            # # 保证原始点不是最大值点，否则返回原点
            # char1 = img_binary[point[1], point[0]]
            # if char1 == 255:
            #     print("初始点已是最大值点，返回原点\n")
            #     point2 = point
            #     point2 = point2[::-1]
        return point2

