#coding:utf-8
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import Qt,QEvent
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2 as cv
import os
from skimage import morphology
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas    # matplotlib画图用
import numpy as np
import json
from lib.hair import getOrientation,get_PCA_angle,getWarpTile,curve_plot,get_width,auto_search,impaint,getMidPoint
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import glob
import time
from math import  *
from screeninfo import get_monitors
from lib.utils import *
from lib.process import MyProcess

process = MyProcess()

import matplotlib.pyplot as plt
# dict转json时需要用到的转码函数
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


#全局变量，图像文件夹路径
imgPath = "data/imgs/"

JOINT = 1
RECT = 2


BINARY = 0
SKELETON = 1



BINARY_NORMAL = 0
BINARY_AUTO = 1
BINARY_DL = 2
BINARY_AUTO_WITH_DL = 3
BINARY_Cluster = 4

class InpaintThread(QThread):  # 步骤1.创建一个线程实例
    mysignal = pyqtSignal(np.ndarray)  # 创建一个自定义信号，元组参数
    def __init__(self,result,image):
        super(InpaintThread, self).__init__()
        self.result_ = result
        self.img = image
    def run(self):

        img_impaint = impaint(self.result_, self.img)
        print('Inpaint over！！！')
        self.mysignal.emit(img_impaint)  # 发射自定义信号


#图像标记类
class Mark(QMainWindow):
    def __init__(self):
        super(Mark, self).__init__()
        uic.loadUi("test.ui",self)

        self.setFixedUI()

        self.roiInf_corrected = None

        self.image_origin = None
        self.image_roi = None
        self.img_binary = None
        self.img_impaint = None

        self.binary_normal = None
        self.binary_auto = None
        self.binary_dl = None
        self.binary_aoto_with_dl = None

        self.result=[]
        self.result_origin=[]
        self.result_bone=[]
        self.handle_bone = False
        self.box_width_init = 20
        self.box_height_init = 60
        self.binary_threshold = 150
        self.img_loaded=False
        self.show_binary=False

        self.magnet_flag = True

        self.plot_alpha = 1.0
        # 模式选择标志
        self.show_type = BINARY
        self.binary_type = BINARY_DL
        self.downsample_ratio = 1
        # 序号
        self.handle_index=-1
        self.impaint_index=0

        self.lengthCorrect = True
        self.isPlot = True

        self.roiInf = [0,0,512]

        self.tmp = 'default'

        self.mousePressFlag = False
        self.initUI()

    # 根据屏幕分辨率设置界面大小
    def setFixedUI(self):
        def findChildrenWidget(widget, allWidgetList):
            if len(widget.children()) > 0:
                for cwidget in widget.children():
                    allWidgetList.append(cwidget)
                    findChildrenWidget(cwidget, allWidgetList)
            return allWidgetList

        m = get_monitors()[0]
        h = int(m.height)
        scalep = h / 2160
        x0 = int(scalep * 2386)
        y0 = int(scalep * 1710)
        self.resize(x0, y0)
        # 缩放UI元素
        widgetList = []
        allWidgetList = []
        allWidgetList = findChildrenWidget(self.centralwidget, allWidgetList)
        for widget in allWidgetList:
            if isinstance(widget, QPropertyAnimation) or "qt_" in widget.objectName() or widget.objectName() == '':
                continue
            widgetList.append(widget)
        for widget in widgetList:
            x = int(widget.geometry().x() * scalep)
            y = int(widget.geometry().y() * scalep)
            w = int(widget.geometry().width() * scalep)
            h = int(widget.geometry().height() * scalep)
            widget.setGeometry(QRect(x, y, w, h))
        self.setFixedSize(self.width(), self.height())

        # pyqtgraph柱状图
        self.plot_widget = PlotWidget(self)
        self.plot_widget.setGeometry(QtCore.QRect(2000* scalep, 1430* scalep, 250* scalep, 250* scalep))
        self.width_count = np.zeros(30)
        self.x = np.arange(30)
        y = np.zeros(30)
        bg = pg.BarGraphItem(x=self.x, y=y, height=0, width=0.8* scalep)
        self.plot_widget.addItem(bg)

        #self.graph.addWidget(self.plot_widget)
        self.graph.addWidget(self.plot_widget)
        self.distinguishValue = self.sliderDistinguish.value()

    def initUI(self):

        # 值的初始化
        self.binary_threshold_normal = self.sliderBinaryNormal.value()
        self.binary_threshold_auto = self.sliderBinaryAuto.value()

        # 加载所有图片名并添加到列表中
        allImgs = glob.glob(imgPath+'*.jpg')

        allImgs += glob.glob(imgPath + '*.png')
        for imgTmp in allImgs:
            self.allFiles.addItem(os.path.basename(imgTmp))   # 将此文件添加到列表中
        self.allFiles.itemClicked.connect(self.itemClick)   #列表框关联时间，用信号槽的写法方式不起作用
 
        # 回调函数
        self.sliderDistinguish.valueChanged.connect(self.distinguishUpdate)

        self.sliderBinaryNormal.valueChanged.connect(self.thresholdBinaryNormalUpdate)
        self.sliderBinaryAuto.valueChanged.connect(self.thresholdBinaryAutoUpdate)


        self.radioBinaryFlag.toggled.connect(self.binaryChecked)
        self.radioSkeletonFlag.toggled.connect(self.skeletonChecked)

        self.editDownsample.textChanged.connect(self.downsampleChanged)


        self.radioBinaryNormal.toggled.connect(self.binaryChecked_Normal)
        self.radioBinaryAuto.toggled.connect(self.binaryChecked_Auto)
        self.radioBinaryDL.toggled.connect(self.binaryChecked_DL)
        self.radioBinaryAutoWithDL.toggled.connect(self.binaryChecked_AutoWithDL)
        self.radioBinaryCluster.toggled.connect(self.binaryChecked_Cluster)




        self.checkLengthCorrect.stateChanged.connect(self.lengthCorrectChecked)
        self.checkPlotFlag.stateChanged.connect(self.plotChecked)

        self.buttonSave.clicked.connect(self.buttonSaveEvent)
        self.buttonImpaint.clicked.connect(self.buttonImpaintEvent)



        
        # self.my_thread.mysignal.connect(self.zhi)  # 自定义信号连接


    def buttonSaveEvent(self):
        curve = np.zeros_like(self.image_origin)
        temp = curve_plot(curve, self.result,distinguishValue=0,color1=(255,255,255),color2=(255,255,255))
        img = temp['img']
       # curve = temp['curve']
        cv.imwrite(self.tmp+'_mask.png',img)
      #  cv.imwrite(self.tmp + '_mask.png', curve)

    def plotBox(self,center,rect_height,rect_width,rect_angle):
        [xp_Click, yp_Click] = center
        ysize, xsize = self.image_origin.shape[:2]
        # 计算四个角点坐标
        p1 = [round(max(0, xp_Click - rect_height // 2)), round(max(0, yp_Click - rect_width // 2))]
        p2 = [round(max(0, xp_Click - rect_height // 2)), round(min(ysize, yp_Click + rect_width // 2))]
        p3 = [round(min(xsize, xp_Click + rect_height // 2)), round(min(ysize, yp_Click + rect_width // 2))]
        p4 = [round(min(xsize, xp_Click + rect_height // 2)), round(max(0, yp_Click - rect_width // 2))]
        xg = np.array([p1[0], p2[0], p3[0], p4[0]])
        yg = np.array([p1[1], p2[1], p3[1], p4[1]])
        xg_t = xp_Click + (xg - xp_Click) * cos(rect_angle) + (yg - yp_Click) * sin(rect_angle)  # 旋转
        yg_t = yp_Click + (yg - yp_Click) * cos(rect_angle) - (xg - xp_Click) * sin(rect_angle)
        cnt = np.array([
            [[int(xg_t[0]), int(yg_t[0])]],
            [[int(xg_t[1]), int(yg_t[1])]],
            [[int(xg_t[2]), int(yg_t[2])]],
            [[int(xg_t[3]), int(yg_t[3])]]
        ])

        # 得到旋转后的box
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        return box


    # 鼠标点击事件
    def eventFilter(self,source, event):
        # 标注图鼠标响应
        if source == self.labelImg or source == self.labelImg_curve:
            # 滚轮————画笔透明度调整
            if event.type() == QEvent.Wheel:
                whell_angle = event.angleDelta()
                if whell_angle.y() > 0:
                    self.plot_alpha+=0.2
                    if self.plot_alpha>1:
                        self.plot_alpha=1
                else:
                    self.plot_alpha -= 0.2
                    if self.plot_alpha < 0:
                        self.plot_alpha = 0
                self.imshow()
            if event.type()==QEvent.MouseButtonPress:
                self.handle_index = -1
                if self.img_loaded == False:
                    return QMainWindow.eventFilter(self, source, event)
                [x, y] = [event.pos().x(), event.pos().y()]

                [x0,x1,y0,y1] = self.roiInf_corrected
                point = [int(x0+x*self.roi_window / self.labelImg.width()), int(y0+y*self.roi_window / self.labelImg.width())]
                if (point[0] > 0 and point[1] > 0):
                    if event.button() == Qt.LeftButton:
                        if self.magnet_flag:
                            point = process.magnet(point,self.getBinary())
                        if self.handle_bone == False:  # 新骨架
                            bone = [point]
                            self.result_bone.append(bone)
                            self.handle_bone = True
                        else:
                            self.result_bone[-1].append(point)
                            self.result.pop(-1)
                            self.result_origin.pop(-1)

                        if len(self.result_bone[-1]) < 2:   # ！！！！
                            mid = self.result_bone[-1][0]
                            self.result.append({'joints': self.result_bone[-1],'width': 1,
                                                'mid': mid})
                            self.result_origin.append(
                                {'joints': self.listModift(self.result_bone[-1]),
                                 'width': 1,
                                 'mid': [mid[0] * self.downsample_ratio, mid[1] * self.downsample_ratio]})
                        else:
                            mid = getMidPoint(self.result_bone[-1])
                            self.result.append({'joints': self.result_bone[-1], 'width': 1,
                                                'mid': mid})
                            self.result_origin.append(
                                {'joints': self.listModift(self.result_bone[-1]),
                                 'width': 1,
                                 'mid': [mid[0] * self.downsample_ratio, mid[1] * self.downsample_ratio]})
                        self.imshow()


                        # # 自动识别框
                        # pkg = auto_search(self.getBinary(), [point[0], point[1]], self.box_width_init, self.box_height_init,
                        #                  is_length_correct=self.lengthCorrect)
                        # # 保存结果
                        # is_find = pkg['is_find']
                        # if (is_find):
                        #     box = pkg['box']
                        #     rect = pkg['rect']
                        #     [[x, y], [w, h], angle] = rect
                        #     x1 = x - w/2 * cos(angle / 57.3)
                        #     y1 = y - w/2 * sin(angle / 57.3)
                        #     x2 = x + w / 2 * cos(angle / 57.3)
                        #     y2 = y + w / 2 * sin(angle / 57.3)
                        #     self.result.append({'joints':[[x1,y1],[x2,y2]], 'rect': rect, 'box': box, 'width': rect[1][1], 'mid':getMidPoint([[x1,y1],[x2,y2]])})
                        # self.imshow()
                    elif event.button() == Qt.RightButton:
                        self.checkThisHair()
                        #self.handle_bone = False
                        min_distance = 99999
                        min_idx = -1
                        for idx, result in enumerate(self.result):
                            joints = result['joints']
                            [x,y]=[0,0]
                            for joint in joints:
                                x += joint[0]
                                y += joint[1]
                            x /= len(joints)
                            y /= len(joints)
                            dis = (point[0] - x) * (point[0] - x) + (point[1] - y) * (point[1] - y)
                            if (dis < min_distance):
                                min_distance = dis
                                min_idx = idx
                        self.handle_index = min_idx
                        self.imshow()
                        pass

                    

            if event.type()==QEvent.MouseMove:
                [x, y] = [event.pos().x(), event.pos().y()]
                if self.roiInf_corrected is not None:
                    [x, y] = [round(x * self.roi_window / self.labelImg.width()),
                                 round(y * self.roi_window / self.labelImg.width())]
                self.imshowBling([x, y])

        # 缩略图鼠标响应
        if source == self.labelImg_roi:
            # 滚轮————缩放ROI
            if event.type() == QEvent.Wheel:
                whell_angle = event.angleDelta()
                if whell_angle.y() > 0:
                    self.roi_window -= self.roi_window*0.1
                else:
                    self.roi_window += self.roi_window*0.1
                self.roi_window = int(max(0, min(self.roi_window,self.image_origin.shape[1]) ))
                # 鼠标滚轮缩放roi框
                self.roiInf = [self.roiInf[0], self.roiInf[1], self.roi_window]
                self.roiUpdate()
                self.imshow()
                self.imshow_small_picture()


            if event.type()==QEvent.MouseButtonPress:
                self.mousePressFlag = True

            elif event.type()==QEvent.MouseButtonRelease:
                self.mousePressFlag = False

            if self.mousePressFlag and event.type()==QEvent.MouseMove:
                [x, y] = [event.pos().x(), event.pos().y()]
                self.roiInf = [x / self.roi_size[0], y / self.roi_size[1], self.roi_window]
                self.roiUpdate()
                self.imshow()
                self.imshow_small_picture()


        return QMainWindow.eventFilter(self, source, event)
    






    def getImage(self,update=False):
        img={
            1: self.image_filter,
            BINARY: self.getBinary(update),
        }
        return img[1]



    # 得专门用一个线程去对整张图进行运算，如果运算完成，则缩略图直接用就行，不需要再进行curve_plot
    def imshowPure(self):   # 权宜之计，之后进一步优化   绘图只在当前窗口上画，然后替换掉原图对应区域————这个已完成，但带来新的问题：缩略图会卡，优化方法见上行
        if self.roiInf_corrected is not None:
            x0, x1, y0, y1 = self.roiInf_corrected
            img = self.label_img[y0:y1, x0:x1, :]

        # 绘制左图
        img = cv.resize(self.label_img, self.img_size)
        qimg = QtGui.QImage(img, img.shape[1], img.shape[0],
                    img.shape[1]*3, QtGui.QImage.Format_RGB888)  # bytesPerLine参数设置为image的width*image.channels
        self.labelImg.setPixmap(QtGui.QPixmap(qimg))
        # 绘制右图
        curve = cv.resize(self.getBinary()[y0:y1, x0:x1], self.img_size)
        qimg = QtGui.QImage(curve, curve.shape[1], curve.shape[0],
                    curve.shape[1]*3, QtGui.QImage.Format_RGB888)  # bytesPerLine参数设置为image的width*image.channels
        self.labelImg_curve.setPixmap(QtGui.QPixmap(qimg))








    # 编辑点的闪烁效果————改成活动的细线标记
    def imshowBling(self,mousePoint):
        if len(self.result)>0 and self.handle_index!=-2:
            if self.handle_bone==False:
                return

            [x,y] = self.result[self.handle_index]['joints'][-1]      # 获取最后一个点

            if self.roiInf_corrected is not None:
                x0, x1, y0, y1 = self.roiInf_corrected

                    # [x, y] = [int(x0 + x * self.roi_window / self.labelImg.width()),
                    #          int(y0 + y * self.roi_window / self.labelImg.width())]

                x = round(x - x0)
                y = round(y - y0)

            temp_img = cv.line(self.label_img.copy(), (x, y), (mousePoint[0], mousePoint[1]), (100, 255, 100), 2, cv.LINE_4)       # 直接用opencv绘制细线

            # temp = curve_plot(self.label_img,[self.result[self.handle_index]], 0,(0, 255, 0),(0, 255, 0),alpha=alpha,handle_diff=[self.roiInf_corrected[0],self.roiInf_corrected[2]])   # 仅对选中的一根毛发进行绘制
            # self.label_img = temp['img']



            img = cv.resize(temp_img, self.img_size)
            qimg = QtGui.QImage(img, img.shape[1], img.shape[0],
                        img.shape[1]*3, QtGui.QImage.Format_RGB888)  # bytesPerLine参数设置为image的width*image.channels
            self.labelImg.setPixmap(QtGui.QPixmap(qimg))







    def imshow(self,update=False,plot=True):
        self.label_img = self.getImage(update)
        temp = curve_plot(self.label_img, self.result, self.distinguishValue/self.downsample_ratio,(255, 0, 0),alpha=self.plot_alpha, roi=self.roiInf_corrected)

        if self.isPlot==True:
            self.label_img = temp['img']   # 用plot后的图
        else:
            if len(self.label_img.shape)==2:
                self.label_img = cv.cvtColor(self.label_img, cv.COLOR_GRAY2BGR)

        if self.roiInf_corrected is not None:
            x0, x1, y0, y1 = self.roiInf_corrected
            # img = self.label_img[y0:y1, x0:x1, :]


        # 将常规绘图与handle_index绘图分开进行，重复调用一次curve_plot
        curve = self.getBinary()[y0:y1, x0:x1]  # 得到二值图ROI
        if len(self.result)>0 and self.handle_index!=-2:
            if self.handle_bone==True:
                alpha=1     # 未标完的毛发，始终呈不透明显示（可能透明化反而好点，存疑）
            else:
                alpha=self.plot_alpha
            temp = curve_plot(self.label_img,[self.result[self.handle_index]], 0,(0, 255, 0),(0, 255, 0),alpha=alpha,handle_diff=[self.roiInf_corrected[0],self.roiInf_corrected[2]])   # 仅对选中的一根毛发进行绘制
            self.label_img = temp['img']
            binary_temp = curve_plot(curve, [self.result[self.handle_index]], 0, (0, 255, 0), (0, 255, 0),
                              alpha=1, handle_diff=[self.roiInf_corrected[0], self.roiInf_corrected[2]], handle_width=2)        # 对二值图进行绘制
            curve = binary_temp['img']



        img = cv.resize(self.label_img, self.img_size)
        qimg = QtGui.QImage(img, img.shape[1], img.shape[0],
                    img.shape[1]*3, QtGui.QImage.Format_RGB888)  # bytesPerLine参数设置为image的width*image.channels
        self.labelImg.setPixmap(QtGui.QPixmap(qimg))
        # 绘制右图
        curve = cv.resize(curve, self.img_size)
        if len(curve.shape)==2:
            qimg = QtGui.QImage(curve, curve.shape[1], curve.shape[0],
                        curve.shape[1]*1, QtGui.QImage.Format_Grayscale8)  # bytesPerLine参数设置为image的width*image.channels
        else:
            qimg = QtGui.QImage(curve, curve.shape[1], curve.shape[0],
                                curve.shape[1] * 3,
                                QtGui.QImage.Format_RGB888)  # bytesPerLine参数设置为image的width*image.channels
        self.labelImg_curve.setPixmap(QtGui.QPixmap(qimg))





        # 绘制统计信息图
        self.width_count = np.zeros(30,dtype=np.uint8)
        for result_ in self.result:
            width = result_['width']
            if(width>30):   # 暂时认为不存在超过30宽度的毛发，后面改成自适应数组
                continue
            self.width_count[int(width)] += 1


        ## matplotlib绘制更好看的柱状图。。。。
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        # num_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        # plt.bar(num_list, self.width_count[:20], color="lightseagreen", tick_label=num_list)
        # for a, b in zip(num_list, self.width_count[:20]):
        #     plt.text(a, b + 0.1, b, ha='center', va='bottom')  # 每个柱顶部显示数值
        # plt.xlabel('宽度')
        # plt.ylabel('数量')


        inf = analysisInf(self.width_count[:self.distinguishValue])
        self.statusNum.setText(inf['num'])
        self.statusMedian.setText(inf['median'])
        self.statusMean.setText(inf['mean'])
        self.statusStd.setText(inf['std'])
        self.statusMode.setText(inf['mode'])
        inf = analysisInf(self.width_count[self.distinguishValue:],offset=self.distinguishValue)
        self.statusNum_2.setText(inf['num'])
        self.statusMedian_2.setText(inf['median'])
        self.statusMean_2.setText(inf['mean'])
        self.statusStd_2.setText(inf['std'])
        self.statusMode_2.setText(inf['mode'])

        # 显示直方图
        self.plot_widget.clear()
        bg1 = pg.BarGraphItem(x=self.x[:self.distinguishValue], height=self.width_count[:self.distinguishValue], width=1.2, brush=pg.mkBrush(color=(200, 0, 0)))
        bg2 = pg.BarGraphItem(x=self.x[self.distinguishValue:], height=self.width_count[self.distinguishValue:], width=1.2, brush=pg.mkBrush(color=(0, 0, 255)))
        self.plot_widget.addItem(bg1)
        self.plot_widget.addItem(bg2)



        json_str = json.dumps(self.result_origin, indent=4, cls=NpEncoder)
        with open(self.tmp + '.json', 'w') as json_file:
            json_file.write(json_str)



    def plotChecked(self, isChecked):
        if (isChecked == 0):
            self.isPlot = False
        else:
            self.isPlot = True
        self.imshow()

    def lengthCorrectChecked(self,isChecked):
        if(isChecked==0):
            self.lengthCorrect = False
        else:
            self.lengthCorrect = True


        #self.imshow()
    def downsampleChanged(self):
        self.downsample_ratio = float(self.editDownsample.text())
        self.editDownsample.clearFocus()
        self.itemClick()
    # 图像显示类型选择
    def skeletonChecked(self,isChecked):
        if isChecked:
            self.show_type = SKELETON
        self.imshow(update=True)
    def binaryChecked(self,isChecked):
        if isChecked:
            self.show_type = BINARY
        self.imshow(update=True)
    # 二值图生成方式选择
    def binaryChecked_Normal(self,isChecked):
        if isChecked:
            self.binary_type = BINARY_NORMAL
            self.getBinary(True)
        self.imshow()
    def binaryChecked_Auto(self,isChecked):
        if isChecked:
            self.binary_type = BINARY_AUTO
            self.getBinary(True)
        self.imshow()
    def binaryChecked_DL(self,isChecked) :
        if isChecked:
            self.binary_type = BINARY_DL
            self.getBinary(True)
        self.imshow()
    def binaryChecked_AutoWithDL(self,isChecked) :
        if isChecked:
            self.binary_type = BINARY_AUTO_WITH_DL
            self.getBinary(True)
        self.imshow()
    def binaryChecked_Cluster(self,isChecked) :
        if isChecked:
            self.binary_type = BINARY_Cluster
            self.getBinary(True)
        self.imshow()



    






    def getBinary(self, update=False):

        if self.binary_type==BINARY_NORMAL:
            if update:
                img_gray = cv.cvtColor(self.image_origin, cv.COLOR_BGR2GRAY)
                _, self.img_binary = cv.threshold(img_gray, self.binary_threshold_normal, 255, cv.THRESH_BINARY_INV)
                if self.show_type == SKELETON:
                    self.skeleton = morphology.skeletonize(self.img_binary / 255).astype(np.uint8) * 255

            if self.show_type==SKELETON:
                return self.skeleton
            else:
                return self.img_binary

        elif self.binary_type==BINARY_AUTO:
            if update:
                img_gray = cv.cvtColor(self.image_origin, cv.COLOR_BGR2GRAY)
                self.binary_auto = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,int(11/self.downsample_ratio),self.binary_threshold_auto)
                if self.show_type == SKELETON:
                    self.skeleton = morphology.skeletonize(self.binary_auto / 255).astype(np.uint8) * 255

            if self.show_type==SKELETON:
                return self.skeleton
            else:
                return self.binary_auto

        elif self.binary_type==BINARY_DL:
            if update:
                img_path = 'data/masks/'+self.tmp.split('/')[-1]+'.png'
                dl_output = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY) # 加载预先生成的mask（后期效果好的话再把模型预测放进来）
                _, self.binary_dl = cv.threshold(dl_output, 50, 255, cv.THRESH_BINARY)
                self.binary_dl = cv.resize(self.binary_dl,
                                (int(self.binary_dl.shape[1] / self.downsample_ratio), int(self.binary_dl.shape[0] / self.downsample_ratio)))
                if self.show_type == SKELETON:
                    self.skeleton = morphology.skeletonize(self.binary_dl / 255).astype(np.uint8) * 255

            if self.show_type==SKELETON:
                return self.skeleton
            else:
                return self.binary_dl

        elif self.binary_type==BINARY_AUTO_WITH_DL:     # 要改成：在线程里对整图进行运算，计算好之后直接读取，而不是每次都再运算一次
            if update:
                img_path = 'data/masks/'+self.tmp.split('/')[-1]+'.png'
                dl_output = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY) 
                _, self.binary_dl = cv.threshold(dl_output, 50, 255, cv.THRESH_BINARY)
                self.binary_aoto_with_dl = process.binary_search(self.image_origin,self.binary_dl)
                if self.show_type==SKELETON:
                    self.skeleton = morphology.skeletonize(self.binary_aoto_with_dl / 255).astype(np.uint8) * 255

            if self.show_type==SKELETON:
                return self.skeleton
            else:
                return self.binary_aoto_with_dl


        elif self.binary_type==BINARY_Cluster:     # 要改成：在线程里对整图进行运算，计算好之后直接读取，而不是每次都再运算一次
            if update:
                img_gray = cv.cvtColor(self.image_origin, cv.COLOR_BGR2GRAY)
                self.binary_cluster = process.binart_cluster(img_gray)
                if self.show_type==SKELETON:
                    self.skeleton = morphology.skeletonize(self.binary_cluster / 255).astype(np.uint8) * 255

            if self.show_type==SKELETON:
                return self.skeleton
            else:
                return self.binary_cluster

        return None





    def distinguishUpdate(self,value):
        self.distinguishValue = int(value)
        self.spinBoxDistinguish.setValue(int(value))
        self.imshow()

    def thresholdBinaryNormalUpdate(self,value):
        self.binary_threshold_normal = int(value)
        self.editThreshold.setText(str(value))


        self.imshow(update=True)

    def thresholdBinaryAutoUpdate(self,value):
        self.binary_threshold_auto = int(value)
        self.editThreshold_2.setText(str(value))

        self.imshow(update=True)


    def buttonImpaintEvent(self):
        print('拔毛函数')
        self.inpaintThread = InpaintThread(self.result, self.image_origin.copy())  # 步骤2. 主线程连接子线
        self.inpaintThread.mysignal.connect(self.buttonImpaintEventSignal)
        self.inpaintThread.start()  # 步骤3 子线程开始执行run函数
        
        # self.img_impaint = impaint(self.result, self.image_origin.copy())
        # self.imshow_small_picture()

    def buttonImpaintEventSignal(self,img_inpaint):
        self.img_impaint = img_inpaint
        self.imshow_small_picture()

        
        # cv.imshow('aaa',img_inpaint)
        # cv.waitKey(0)







    def imshow_small_picture(self):
        curve = self.img_impaint.copy()
        x0,x1,y0,y1 = self.roiInf_corrected
        cv.rectangle(curve, (x0,y0), (x1,y1), (0,0,255), int(self.image_origin.shape[0]/128))
        curve = cv.resize(curve, self.roi_size)
        curve = QtGui.QImage(curve, curve.shape[1], curve.shape[0],
                             curve.shape[1] * 3,
                             QtGui.QImage.Format_RGB888)  # bytesPerLine参数设置为image的width*image.channels
        self.labelImg_roi.setPixmap(QtGui.QPixmap(curve))


    def roiUpdate(self):
        x,y,w = self.roiInf
        x = x * self.image_origin.shape[1]
        y = y * self.image_origin.shape[0]
        h = int(w / 2 /self.widthHeightRatio)
        w = int(w / 2)

        x =  max(w, min(x, self.image_origin.shape[1] - w))
        y = max(h, min(y, self.image_origin.shape[0] - h))
        x0 = int(x - w)
        x1 = int(x + w)
        y0 = int(y - h)
        y1 = int(y + h)

        self.image_roi = self.image_origin[y0:y1,x0:x1,:].copy()
        self.roiInf_corrected = [x0,x1,y0,y1]





    # 列表图片选取
    def itemClick(self):  #列表框单击事件
        self.handle_index = -2

        # self.distinguishValue = 0   # 二分类宽度阈值


        self.tmp = imgPath + self.allFiles.currentItem().text()  #图像的绝对路径
        src = cv.imread(str(self.tmp),1)      #读取图像
        self.widthHeightRatio = src.shape[1]/src.shape[0]

        if src.shape[0]<1024:
            self.downsample_ratio = 1
        src = cv.resize(src,(int(src.shape[1]/self.downsample_ratio),int(src.shape[0]/self.downsample_ratio)))

        self.editDownsample.setText(str(self.downsample_ratio))
        # 加载标注文件
        self.result, self.result_origin = self.loadJson(self.tmp + '.json')


        src = cv.cvtColor(src, cv.COLOR_BGR2RGB)
        self.image_filter = cv.medianBlur(src, 3)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
        self.image_filter = cv.filter2D(self.image_filter, -1, kernel=kernel)


        #self.image_filter = src





        self.image_origin = src.copy()
        # 怎么在修改分辨率时不重新加载？？
        self.img_impaint = src.copy()

        # 缩略图展示
        self.roi_size = (self.labelImg_roi.width(), self.labelImg_roi.height())
        self.roi_window = 512
        self.roiInf = [0.5,0.5,self.roi_window]
        self.roiUpdate()
        self.imshow_small_picture()


        self.img_size=(self.labelImg.width(), self.labelImg.height())

        #labelImg_roi

        self.img_loaded=True
        self.allFiles.clearFocus()



        self.getBinary(update=True)


        self.imshow()



    def checkThisHair(self):
        if self.handle_bone == True:
            self.handle_bone = False
            self.result.pop(-1)
            self.result_origin.pop(-1)
            if len(self.result_bone[-1]) >= 2:
                joint_temp = self.result_bone[-1]

                joint_temp = process.border(joint_temp,self.getBinary())
                height = process.waist(joint_temp,self.getBinary())

                mid = getMidPoint(self.result_bone[-1])
                self.result.append({'joints': self.result_bone[-1], 'width': height, 'mid': mid})
                self.result_origin.append({'joints': self.listModift(self.result_bone[-1]), 'width': int(height*self.downsample_ratio),
                                    'mid': [mid[0]*self.downsample_ratio, mid[1]*self.downsample_ratio]})

    def generateHairPath(self):
        if self.handle_bone == True:
            self.handle_bone = False
            if len(self.result_bone[-1]) >= 2:
                joint_temp = self.result_bone[-1]

                joint_temp = process.border(joint_temp,self.getBinary())
                x0, x1 = min(joint_temp[0][0], joint_temp[-1][0]), max(joint_temp[0][0], joint_temp[-1][0])
                y0, y1 = min(joint_temp[0][1], joint_temp[-1][1]), max(joint_temp[0][1], joint_temp[-1][1])
                #[x0,y0],[x1,y1] = joint_temp[0],joint_temp[-1]
                x0_outer = x0 - 30
                x1_outer = x1 + 30
                y0_outer = y0 - 30
                y1_outer = y1 + 30



                img_binary = self.getBinary()[y0_outer:y1_outer, x0_outer:x1_outer]  # 得到二值图ROI
                #img_binary = self.skeleton[y0_outer:y1_outer, x0_outer:x1_outer]  # 得到二值图ROI
                joints,length = process.generate_path(img_binary,[30,30],[(y1-y0)+30,x1-x0+30])
                for joint in joints:
                    joint[0] = joint[0] - 30 + x0
                    joint[1] = joint[1] - 30 + y0
                if len(joints)==0:
                    return

                self.result.pop(-1)
                self.result_origin.pop(-1)

                height = process.waist(joints,self.getBinary())
                mid = getMidPoint(joints)

                self.result.append({'joints': joints, 'width': height, 'mid': mid})
                self.result_origin.append({'joints': self.listModift(joints), 'width': int(height*self.downsample_ratio),
                                    'mid': [mid[0]*self.downsample_ratio, mid[1]*self.downsample_ratio]})

           


    # 捕捉键盘事件
    def keyReleaseEvent(self, QKeyEvent):
        if QKeyEvent.key() == Qt.Key_E:
            self.magnet_flag=True

    def keyPressEvent(self, QKeyEvent):
        if self.img_loaded==False:
            return


 
        if QKeyEvent.key() == Qt.Key_E:
            self.magnet_flag=False
        else:
            self.magnet_flag=True

        # 确认
        if QKeyEvent.key() == Qt.Key_E:
            self.checkThisHair()
            self.imshow()
            return
        elif QKeyEvent.key() == Qt.Key_Q:
            begin = time.time()
            self.generateHairPath()
            begin1 = time.time()
            self.imshow()
            return




        #参数1  控件
        if(len(self.result)<1):
            return
        if QKeyEvent.key()== Qt.Key_Backspace:  # 删除
            self.handle_bone = False
            self.result.pop(self.handle_index)
            self.result_origin.pop(self.handle_index)

            if len(self.result)==0:
                self.handle_index = -2
            else:
                self.handle_index = -1
            self.imshow()


        joints = self.result[self.handle_index]['joints']
        width = self.result[self.handle_index]['width']
        mid = self.result[self.handle_index]['mid']

        #rect=self.result[self.handle_index]['rect']
       # [[x,y],[w,h],angle] = rect

        begin = time.time()
        if QKeyEvent.key() == Qt.Key_A:  # 左移
            for joint in joints:
                joint[0] -= 1
            mid = getMidPoint(joints)
            self.result.pop(self.handle_index)
            self.result_origin.pop(self.handle_index)
            self.result.append({'joints':joints, 'width':width,'mid':mid})
            self.result_origin.append(
                {'joints': self.listModift(joints), 'width': int(width*self.downsample_ratio),
                 'mid': [mid[0] * self.downsample_ratio, mid[1] * self.downsample_ratio]})
            self.handle_index = -1      # 修正后，对应目标的序号必定变为-1
            self.imshow()


        elif QKeyEvent.key()== Qt.Key_D:  # 右移
            for joint in joints:
                joint[0] += 1
            mid = getMidPoint(joints)
            self.result.pop(self.handle_index)
            self.result_origin.pop(self.handle_index)
            self.result.append({'joints':joints, 'width':width,'mid':mid})
            self.result_origin.append(
                {'joints': self.listModift(joints), 'width': int(width*self.downsample_ratio),
                 'mid': [mid[0] * self.downsample_ratio, mid[1] * self.downsample_ratio]})
            self.handle_index = -1  
            self.imshow()


        elif QKeyEvent.key()== Qt.Key_S:  # 下移
            for joint in joints:
                joint[1] += 1
            mid = getMidPoint(joints)
            self.result.pop(self.handle_index)
            self.result_origin.pop(self.handle_index)
            self.result.append({'joints':joints, 'width':width,'mid':mid})
            self.result_origin.append(
                {'joints': self.listModift(joints), 'width': int(width*self.downsample_ratio),
                 'mid': [mid[0] * self.downsample_ratio, mid[1] * self.downsample_ratio]})
            self.handle_index = -1  
            self.imshow()

        elif QKeyEvent.key()== Qt.Key_W:  # 上移
            for joint in joints:
                joint[1] -= 1
            mid = getMidPoint(joints)
            self.result.pop(self.handle_index)
            self.result_origin.pop(self.handle_index)
            self.result.append({'joints':joints, 'width':width,'mid':mid})
            self.result_origin.append(
                {'joints': self.listModift(joints), 'width': int(width*self.downsample_ratio),
                 'mid': [mid[0] * self.downsample_ratio, mid[1] * self.downsample_ratio]})
            self.handle_index = -1  
            self.imshow()


        elif QKeyEvent.key()== Qt.Key_Up:  # 变宽
            width += 1
            self.result.pop(self.handle_index)
            self.result_origin.pop(self.handle_index)
            self.result.append({'joints':joints, 'width':width,'mid':mid})
            self.result_origin.append(
                {'joints': self.listModift(joints), 'width': int(width*self.downsample_ratio),
                 'mid': [mid[0] * self.downsample_ratio, mid[1] * self.downsample_ratio]})
            self.handle_index = -1  
            self.imshow()

        elif QKeyEvent.key()== Qt.Key_Down:   # 变窄
            width -= 1
            self.result.pop(self.handle_index)
            self.result_origin.pop(self.handle_index)
            self.result.append({'joints':joints,  'width':width,'mid':mid})
            self.result_origin.append(
                {'joints': self.listModift(joints), 'width': int(width*self.downsample_ratio),
                 'mid': [mid[0] * self.downsample_ratio, mid[1] * self.downsample_ratio]})
            self.handle_index = -1  
            self.imshow()
        if len(joints)==2:
            pass
            # if QKeyEvent.key()== Qt.Key_Left:   # 逆时针旋转
            #     angle -= 2
            #     rect = ((x,y),(w,h),angle)
            #     box = cv.boxPoints(rect)
            #     box = np.int0(box)
            #     self.result.pop(self.handle_index)
            #     self.result.append({'type':type, 'joints':joints, 'rect':None, 'box':None, 'width':width})
            #     self.handle_index = -1
            #     self.imshow()
            #
            # if QKeyEvent.key()== Qt.Key_Right:  # 顺时针旋转
            #     angle += 2
            #     rect = ((x,y),(w,h),angle)
            #     box = cv.boxPoints(rect)
            #     box = np.int0(box)
            #     self.result.pop(self.handle_index)
            #     self.result.append({'type':type, 'joints':joints,'rect':rect, 'box':box, 'width':rect[1][1]})
            #     self.handle_index = -1
            #     self.imshow()





        if QKeyEvent.modifiers() == Qt.ControlModifier|Qt.ShiftModifier and QKeyEvent.key() == Qt.Key_A:  # 三键组合
            print('按下了Ctrl+Shift+A键')


    def listModift(self,joints):
        joints_origin = []
        for joint in joints:
            joints_origin.append([int(joint[0] * self.downsample_ratio),int(joint[1] * self.downsample_ratio)])
        return joints_origin

    # 加载json文件，并转换格式
    def loadJson(self,jsonPath):
        result = []
        result_origin =[]
        if os.path.exists(jsonPath):
            with open(jsonPath, 'r') as f:
                datas = json.load(f)

            for data in datas:
                if len(data['joints'])<2:
                    continue
                d={}
                d_origin={}
                joints = data['joints']
                joints_origin = []
                for joint in joints:
                    joints_origin.append(joint.copy())
                    joint[0] = int(joint[0]/self.downsample_ratio)
                    joint[1] = int(joint[1]/self.downsample_ratio)
                d['joints'] = joints
                d_origin['joints'] = joints_origin


                d_origin['width'] = data['width']
                d['width'] = int(data['width']/self.downsample_ratio)


                if 'mid' in data.keys():
                    d_origin['mid'] = data['mid'].copy()
                    mid = data['mid']
                    mid[0] = int(mid[0]/self.downsample_ratio)
                    mid[1] = int(mid[1]/self.downsample_ratio)
                    d['mid'] = mid
                else:
                    d['mid'] = getMidPoint(data['joints'])
                result.append(d)
                result_origin.append(d_origin)
        return result,result_origin
    

if  __name__ == "__main__":                                 # main函数
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = Mark()
    MainWindow.show()
    app.installEventFilter(MainWindow)
    sys.exit(app.exec_())