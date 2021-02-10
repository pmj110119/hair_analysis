import cv2
import numpy as np
"""
magnet:
输入：
point：np.array([x,y]);图片点的横纵坐标位置 x:横轴；y:纵轴；
img ： 需要点点的图片的灰度图；

return：
point2: 吸引之后的点的坐标，
"""
# img = cv2.imread("D:/desktop\img/117CCBBA.jpg", 0)
# 展示一张图片，选择点，并用标出该点的坐标位置，
# 自动在图像中标出吸铁石之后的点及其坐标位置
# img[680:690,610] = 0
# cv2.imshow("img",img)
# cv2.waitKey(0)
# binarize
# img_bi = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,101,30)

def magnet(point,img):
    """
    point: the loc of the raw point
    img:the gray img
    return point2: the calculated point
    """
    # 试图找到输入点上下size个像素垂直线中，第一个和最后一个值最小的点的坐标，之后求均值得到中点
    # 设置线长，灰度图片

    # point = [680,610]

    size = 16
    # 二值化
    # patch_img = img[point[0] - size:point[0] + size + 1, point[1] - size:point[1] + size + 1]
    # patch_img = patch_img.astype(float)
    img_bi = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,101,30)

    # 确定直线矩阵
    line_cloumn = img_bi[point[0]-size:point[0]+size+1,point[1]]
    line_row = img_bi[point[0],point[1]-size:point[1]+size+1]
    # print(line)

    # 确定垂直线第一个和最后一个数值最低的点的坐标
    indx = cv2.minMaxLoc(line_cloumn,None)
    indx2 = np.where(line_cloumn == indx[0])
    l0 = indx2[0] #所有最小值点的坐标
    l = np.size(indx2,1)  #最小值的个数
    firstmin_loc = indx[2]
    firstmin_loc = np.array([firstmin_loc[1],0])
    lastmin_loc =np.array([l0[l-1],0])
    # 确定中间点的坐标
    mid = (firstmin_loc+lastmin_loc)/2
    mid = np.around(mid)
    if mid[0] == size:
        mid = firstmin_loc
    row = mid[0]-size
    point2_column = np.array([point[0]+row,point[1]])

    #确定水平线第一个和最后一个数值最低的点的坐标
    indx3 = cv2.minMaxLoc(line_row, None)
    indx4 = np.where(line_row == indx3[0])
    l01 = indx4[0]  # 所有最小值点的坐标
    l11 = np.size(indx4, 1)  # 最小值的个数
    firstmin_loc1 = indx3[2]
    firstmin_loc1 = np.array([0,firstmin_loc1[1]])
    lastmin_loc1 = np.array([0,l01[l11 - 1]])
    # 确定中间点的坐标
    mid1 = (firstmin_loc1 + lastmin_loc1) / 2
    mid1 = np.around(mid1)
    if mid1[1] == size:
        mid1 = firstmin_loc1
    row1 = mid1[1] - size
    point2_column1 = np.array([point[0], point[1]+row1])
    if abs(row) <= abs(row1):
        point2 = point2_column
    else:
        point2 = point2_column1

    return point2




#
# def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # x表示图片的列， y表示图片的行
#         xy = "%d,%d" % (y, x)
#         print("你点的位置：\n",xy)
#         cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
#         cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                     1.0, (0, 0, 0), thickness=1)
#
#         point3 = [y, x]
#         point4 = magnet(point3, img)
#         # x1表示图片的行，y1表示图片的列
#         x1 = point4[0]
#         x1 = x1.astype(np.int)
#         y1 = point4[1]
#         y1 = y1.astype(np.int)
#         x1y1 = "%d,%d" % (x1, y1)
#         print("吸过的位置：\n", x1y1)
#         x2 = x1 - x
#         y2 = y1 - y
#         x2y2 = "%d,%d" % (y2, x2)
#         print("吸过位置-原位置：\n", x2y2)
#         # print("吸引位置灰度-原位置灰度：\n", img[x,y]-img[x1,y1])
#         cv2.circle(img, (y1, x1), 1, (255, 0, 0), thickness=-1)
#         cv2.putText(img, x1y1, (y1, x1), cv2.FONT_HERSHEY_PLAIN,
#                     1.0, (0, 0, 0), thickness=1)
#         cv2.imshow("image", img)
#
#
#
#
#
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
# cv2.imshow("image", img)
#
# while (True):
#     try:
#         cv2.waitKey(100)
#     except Exception:
#         cv2.destroyWindow("image")
#         break
#
# cv2.waitKey(0)
# cv2.destroyAllWindow()





