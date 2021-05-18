
#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..
#                     美女保佑 永无BUG
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import Qt,QEvent
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import qdarkstyle
import sys
import cv2 as cv
import os
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas    # matplotlib画图用
import numpy as np
import json
from lib.hair import getOrientation,get_PCA_angle,getWarpTile,curve_plot,endpoint_plot,auto_search,impaint,getMidPoint
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import glob
import time
import pandas as pd
import csv

from math import  *
from screeninfo import get_monitors
from lib.utils import *
from lib.process import MyProcess
from lib.authorList import AuthorWindow

process = MyProcess()

import matplotlib.pyplot as plt


DEBUG = False






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




BINARY_NORMAL = 0
BINARY_AUTO = 1
BINARY_DL = 2
BINARY_INNER = 3


HANDLE_Q = 0
HANDLE_E = 1
HANDLE_INNER = 2

class InpaintThread(QThread):  # 步骤1.创建一个线程实例
    mysignal = pyqtSignal(np.ndarray)  # 创建一个自定义信号，元组参数
    def __init__(self,result,image):
        super(InpaintThread, self).__init__()
        self.result_ = result
        self.img = image
    def run(self):
        img_impaint = impaint(self.result_, self.img)
        self.mysignal.emit(img_impaint)  # 发射自定义信号


class FullImagePlotThread(QThread):  # 步骤1.创建一个线程实例
    signal_img = pyqtSignal(np.ndarray)  # 创建一个自定义信号，元组参数
    def __init__(self):
        super(FullImagePlotThread, self).__init__()
        self.updateFlag = False
    def update(self,result,image,distinguishValue,plot_alpha):
        self.result = result
        self.img = image
        self.distinguishValue = distinguishValue
        self.plot_alpha = plot_alpha
        self.img_ploted = image
        self.updateFlag = True
    def run(self):
        while True:
            img = curve_plot(self.img, self.result, self.distinguishValue, (255, 0, 0),alpha=self.plot_alpha)
            if self.updateFlag:
                self.signal_img.emit(img)  # 发射自定义信号
                self.updateFlag = False
            self.msleep(1000)

#torch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# class KeypointLoadThread(QThread):
#     model_signal = pyqtSignal(list)
#     def __init__(self,model_path):
#         super(KeypointLoadThread, self).__init__()
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         self.device = torch_device
#         self.model_path = model_path
#     def run(self):
#         self.model = unet_resnet('resnet34', 3, 1, False).to(self.device)
#         pre = torch.load(self.model_path, map_location=self.device)
#         self.model.load_state_dict(pre)
#         self.model_signal.emit([self.model])  # 发射自定义信号








class BinaryWindow(QMainWindow):
    def __init__(self):
        super(BinaryWindow, self).__init__()
        uic.loadUi("assets/binary.ui", self)

#图像标记类
class Mark(QMainWindow):
    def __init__(self):
        super(Mark, self).__init__()
        uic.loadUi("assets/test.ui",self)




        self.setMenu()
        self.setFixedUI()
        self.fullImagePlotThread = FullImagePlotThread()
        self.roiInf_corrected = None

        self.image_origin = None
        self.image_roi = None
        self.img_binary = None
        self.img_impaint = None
        self.img_plot = None
        self.qmutex_img_plot = QMutex()

        self.binary_normal = None
        self.binary_auto = None
        self.binary_dl = None
        self.binary_inner = None

        self.result=[]
        self.autoDetectThread = None

        self.result_bone=[]
        self.endpoints = []
        self.handle_bone = False
        self.inner_stack = []
        self.box_width_init = 20
        self.box_height_init = 60
        self.binary_threshold = 150
        self.img_loaded=False
        self.show_binary=False

        self.magnet_flag = True

        self.plot_alpha = 1.0
        self.autoDetectRatio = 100
        # 模式选择标志
        self.show_type = BINARY
        self.binary_type = BINARY_AUTO
        self.downsample_ratio = 1
        # 序号
        self.handle_index=-1
        self.impaint_index=0

        self.lengthCorrect = True
        self.isPlot = True

        self.roiInf = [0,0,512]

        self.tmp = 'default'

        self.mousePressFlag = False
        self.autoRoiPressFlag = False

        self.handleMode = 0

        self.segmentation_model = None
        self.waitForSegmentation = False



        self.initUI()



    # 根据屏幕分辨率设置界面大小
    def setFixedUI(self):
        m = get_monitors()[0]
        h = int(m.height)
        self.scalepFix = h / 1080
        self.scalep = 0.85 * self.scalepFix

        self.fontSize = 18

        self.setResolution(self.scalep)

    def batch_operation(self):
            directory = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", ".")
            imgs = glob.glob(directory+'/*.jpg')
            print(directory)


    def setMenu(self):
        self.menubar = self.menuBar()  # 获取窗体的菜单栏

        self.menu = self.menubar.addMenu(" 文件 ")
        # # 批处理
        # self.batchMenu = QAction("批处理", self)
        # self.batchMenu.triggered.connect(self.batch_operation)
        # self.menu.addAction(self.batchMenu)

        # 二值化调整
        self.binaryMenu = QAction("二值图调整", self)
        self.binaryMenu.setShortcut("Ctrl+b")  # 设置快捷键
        self.binaryMenu.triggered.connect(self.showBinaryMenu)
        self.menu.addAction(self.binaryMenu)

        # 展示好看的柱状图
        self.showBar = QAction("生成柱状图", self)
        self.showBar.setShortcut("Ctrl+p")  # 设置快捷键
        self.showBar.triggered.connect(self.showMatplotlibBar)
        self.menu.addAction(self.showBar)
        # 导出结果表
        self.exportCSV = QAction("导出csv", self)
        self.exportCSV.setShortcut("Ctrl+s")  # 设置快捷键
        self.exportCSV.triggered.connect(self.buttonSaveEvent)
        self.menu.addAction(self.exportCSV)
        #self.file.triggered[QAction].connect(self.processtrigger)
        # # 导出二值图
        # self.binarySave = QAction("导出二值图", self)
        # self.binarySave.setShortcut("Ctrl+a")  # 设置快捷键
        # self.binarySave.triggered.connect(self.saveBinary)
        # self.menu.addAction(self.binarySave)




        # #self.menu = self.menubar.addMenu("菜单")
        # # 调整分辨率
        # self.resolution = self.menubar.addMenu(" 分辨率 ")
        # resolution70Action = QAction(' 70% ', self)
        # resolution70Action.triggered.connect(self.resolution70)
        # self.resolution.addAction(resolution70Action)  # Edit下这是copy子项
        # resolution80Action = QAction(' 80% ', self)
        # resolution80Action.triggered.connect(self.resolution80)
        # self.resolution.addAction(resolution80Action)  # Edit下这是copy子项
        # resolution90Action = QAction(' 90% ', self)
        # resolution90Action.triggered.connect(self.resolution90)
        # self.resolution.addAction(resolution90Action)  # Edit下这是copy子项
        # resolution100Action = QAction(' 100% ', self)
        # resolution100Action.triggered.connect(self.resolution100)
        # self.resolution.addAction(resolution100Action)  # Edit下这是copy子项

        # 帮助
        self.helpMenu = self.menubar.addMenu(" 帮助 ")
        self.help = QAction(" 使用说明 ", self)
        self.help.setShortcut("Ctrl+h")  # 设置快捷键
        self.help.triggered.connect(self.showHelpDialog)
        self.helpMenu.addAction(self.help)
        # 作者名单
        self.author = QAction(" 贡献者名单 ", self)
        self.author.triggered.connect(self.showAuthorList)
        self.helpMenu.addAction(self.author)

    def showBinaryMenu(self):
        self.binaryWindow = BinaryWindow()
        self.binaryWindow.show()
        # 二值化方法滑块
        self.binaryWindow.sliderBinaryNormal.valueChanged.connect(self.thresholdBinaryNormalUpdate)
        self.binaryWindow.sliderBinaryAuto.valueChanged.connect(self.thresholdBinaryAutoUpdate)
        # 二值化方法
        self.binaryWindow.radioBinaryNormal.toggled.connect(self.binaryChecked_Normal)
        self.binaryWindow.radioBinaryAuto.toggled.connect(self.binaryChecked_Auto)
        self.binaryWindow.radioBinaryDL.toggled.connect(self.binaryChecked_DL)
        self.binaryWindow.radioBinaryInner.toggled.connect(self.binaryChecked_Inner)
        # 闭运算
        self.binaryWindow.sliderBinaryClose.valueChanged.connect(self.binaryCloseUpdate)


    def showHelpDialog(self):
        dialog = QDialog()
        word = QLabel("------------------------帮助页面----------------------------------\n"
                      "\n 1. 左侧列表单击选择图片"
                      "\n 2. 鼠标在右上角的缩略图中单击、滑动滚轮，进行图像移动和放缩"
                      "\n 3. 自动标注"
                      "\n   - 点击“自动识别”按钮，即可在当前ROI区域内进行毛发的自动识别"
                      "\n 4. 手动质检"
                      "\n   - 选择标注模式"
                      "\n   - 鼠标左击，标注端点（或内点），Q键识别"
                      "\n   - 右键选中最近的毛发，WASD调整位置、方向键 ↑ ↓ 调整宽度"
                      "\n   - E键删除选中的毛发"
                      "\n 5. 自动提取宽度不准时的做法"
                      "\n   - 查看二值图，切换二值图方式或调整阈值"
                      "\n   - 方向键手动调整宽度"
                      "\n 6. 单击“拔毛”按钮，已标注的毛发在右上角缩略图中会消失。"
                      "\n   - 每完成一部分毛发标注后，通过“拔毛”来检查是否有漏标注",
                      dialog)
        dialog.setGeometry(QRect(500, 500, 500, 400))
        # btn = QPushButton("已导出结果至 result.csv", dialog)
        # btn.move(50, 50)
        dialog.setWindowTitle("message")
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.exec_()


    def showAuthorList(self):
        self.authorWindow = AuthorWindow()
        self.authorWindow.show()


    def saveBinary(self):
        binary = curve_plot(np.zeros_like(self.getBinary()), self.result, 0, (255, 255, 255), (255, 255, 255), alpha=1)  # 对二值图进行绘制
        print('data/dataset/full_masks/'+os.path.basename(self.tmp)+'.png')
        cv.imwrite('data/dataset/full_masks/'+os.path.basename(self.tmp).split('.')[0]+'.png',binary)


    def initUI(self):

        # 值的初始化
        self.binary_threshold_normal = 140
        self.binary_threshold_auto = 5
        self.binary_close = 3
        # 加载所有图片名并添加到列表中
        allImgs = glob.glob(imgPath+'*.jpg')

        allImgs += glob.glob(imgPath + '*.png')
        for imgTmp in allImgs:
            self.allFiles.addItem(os.path.basename(imgTmp))   # 将此文件添加到列表中
        self.allFiles.itemClicked.connect(self.itemClick)   #列表框关联时间，用信号槽的写法方式不起作用

        self.sliderDistinguish.valueChanged.connect(self.distinguishUpdate)

        # self.radioBinaryAutoWithDL.toggled.connect(self.binaryChecked_AutoWithDL)
        # self.radioBinaryCluster.toggled.connect(self.binaryChecked_Cluster)

        # 手动标注模式的选择
        self.radioHangleModeAuto.toggled.connect(self.handleModeChanged_Auto)
        self.radioHangleModeHandleQ.toggled.connect(self.handleModeChanged_HandleQ)
        self.radioHangleModeHandleE.toggled.connect(self.handleModeChanged_HandleE)



        self.checkMagnetFlag.stateChanged.connect(self.magnetChecked)

     

        self.buttonImpaint.clicked.connect(self.buttonImpaintEvent)



        self.innerClear()


        self.timer = QTimer()
        self.timer.start(300)  # 每过300ms，更新一次绘图线程的值
        self.timer.timeout.connect(self.fullImagePlotThread_update)

    def fullImagePlotThread_update(self):
        self.fullImagePlotThread.update(self.result, self.image_origin, self.distinguishValue, self.plot_alpha)



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
                    if self.plot_alpha>1.0:
                        self.plot_alpha=1.0
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
                        if self.handleMode == HANDLE_INNER:
                            self.handle_index = -2
                            if self.roiInf_corrected is not None:
                                x0, x1, y0, y1 = self.roiInf_corrected
                                x_clip_min = x0
                                x_clip_max = x1
                                y_clip_min = y0
                                y_clip_max = y1
                            else:
                                x_clip_min = 0
                                x_clip_max = self.image_origin.shape[1]-1
                                y_clip_min = 0
                                y_clip_max = self.image_origin.shape[0]-1

                            x_min = np.clip(point[0] - 100, x_clip_min, x_clip_max)
                            x_max = np.clip(point[0] + 100, x_clip_min, x_clip_max)
                            y_min = np.clip(point[1] - 100, y_clip_min, y_clip_max)
                            y_max = np.clip(point[1] + 100, y_clip_min, y_clip_max)

                            inner_binary = process.innerBinary(self.image_origin[y_min:y_max,x_min:x_max], [point[0]-x_min, point[1]-y_min])
                            self.inner_stack.append([[x_min,x_max,y_min,y_max], inner_binary])

                            self.imshow(inner_handle=True)

                        else:
                            if self.magnet_flag:    # 吸铁石
                                point = process.magnet(point,self.getBinary())
                                point[0] = np.clip(point[0],0,self.image_origin.shape[1]-1)
                                point[1] = np.clip(point[1],0,self.image_origin.shape[0]-1)

                            if self.handle_bone == False:  # 新骨架
                                bone = [point]
                                self.result_bone.append(bone)
                                self.handle_bone = True
                            else:
                                self.result_bone[-1].append(point)
                                self.result.pop(-1)


                            if len(self.result_bone[-1]) < 2:   # ！！！！
                                mid = self.result_bone[-1][0]
                                self.result.append({'joints': self.result_bone[-1],'width': 1,
                                                    'mid': mid})

                            else:
                                mid = getMidPoint(self.result_bone[-1])
                                self.result.append({'joints': self.result_bone[-1], 'width': 1,
                                                    'mid': mid})

                            self.imshow()

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


            #         elif event.button() == Qt.MiddleButton:
            #             self.autoRoiPressFlag = True
            #             self.autoRoi = [point[0], point[1], 0, 0]
            #
            # if event.type() == QEvent.MouseButtonRelease and self.autoRoiPressFlag == True:
            #     self.autoRoiPressFlag = False
            #     self.autoRoi[2:4] = [point[0], point[1]]
            #
            # if self.autoRoiPressFlag and event.type() == QEvent.MouseMove:
            #     self.autoRoi[2:4] = [point[0], point[1]]




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
                x0, x1, y0, y1 = self.roiInf_corrected
                self.setImg2Label(self.img_plot[y0:y1, x0:x1], self.getBinary()[y0:y1, x0:x1])
                self.imshow_small_picture()



            if event.type()==QEvent.MouseButtonPress:
                self.mousePressFlag = True

            elif event.type()==QEvent.MouseButtonRelease:
                self.mousePressFlag = False

            if self.mousePressFlag and event.type()==QEvent.MouseMove:
                [x, y] = [event.pos().x(), event.pos().y()]
                self.roiInf = [x / self.roi_size[0], y / self.roi_size[1], self.roi_window]
                self.roiUpdate()
                x0, x1, y0, y1 = self.roiInf_corrected
                self.setImg2Label(self.img_plot[y0:y1, x0:x1], self.getBinary()[y0:y1, x0:x1])
                self.imshow_small_picture()





        return QMainWindow.eventFilter(self, source, event)
    






    def getImage(self,update=False):
        img={
            1: self.image_origin,
            BINARY: self.getBinary(update),
        }
        return img[1]






    # 编辑点的闪烁效果————改成活动的细线标记
    def imshowBling(self,mousePoint):
        if len(self.result)>0 and self.handle_index!=-2:
            if self.handle_bone==False:
                return

            [x,y] = self.result[self.handle_index]['joints'][-1]      # 获取最后一个点


            x0, x1, y0, y1 = self.roiInf_corrected
            x = round(x - x0)
            y = round(y - y0)

            temp_img = cv.line(self.img_plot[y0:y1,x0:x1].copy(), (x, y), (mousePoint[0], mousePoint[1]), (100, 255, 100), 2, cv.LINE_4)       # 直接用opencv绘制细线

            img = cv.resize(temp_img, self.img_size)
            qimg = QtGui.QImage(img, img.shape[1], img.shape[0],
                        img.shape[1]*3, QtGui.QImage.Format_RGB888)  # bytesPerLine参数设置为image的width*image.channels
            self.labelImg.setPixmap(QtGui.QPixmap(qimg))


    def imshowSingleUpdate(self,single_result):
        x0, x1, y0, y1 = [0,9999,0,9999]
        for result_ in single_result:
            joints = result_['joints']
            for x,y in joints:
                if x < x0:
                    x0 = x
                elif x > x1:
                    x1 = x
                if y < y0:
                    y0 = y
                elif y>y1:
                    y1 = y
        label_img = curve_plot(self.img_plot, single_result, self.distinguishValue/self.downsample_ratio,(255, 0, 0),alpha=self.plot_alpha, roi=[x0, x1, y0, y1])
        self.img_plot[y0:y1, x0:x1] = label_img
        [x0, x1, y0, y1] = self.roiInf_corrected



        if len(self.result) > 0 and self.handle_index != -2:

            label_img = curve_plot(self.img_plot[y0:y1, x0:x1], [self.result[self.handle_index]], 0, (0, 255, 0), (0, 255, 0),
                                        alpha=self.plot_alpha, handle_diff=[self.roiInf_corrected[0],
                                                                            self.roiInf_corrected[2]])  # 仅对选中的一根毛发进行绘制
            curve = self.getBinary()[y0:y1, x0:x1]  # 得到二值图ROI
            curve = curve_plot(curve, [self.result[self.handle_index]], 0, (0, 255, 0), (0, 255, 0),
                               alpha=1, handle_diff=[self.roiInf_corrected[0], self.roiInf_corrected[2]],
                               handle_width=2)  # 对二值图进行绘制

            self.setImg2Label(label_img, curve)

        else:
            self.setImg2Label(self.img_plot[y0:y1, x0:x1])




    def imshow(self,update=False,plot=True, inner_handle=True, jump=False):

        x0, x1, y0, y1 = self.roiInf_corrected
        if jump==False:
            self.label_img = self.getImage(update).copy()
        else:
            self.label_img = self.img_plot[y0:y1, x0:x1]

        if inner_handle:
            for [x_min,x_max,y_min,y_max],inner_binary in self.inner_stack:
                roi_img = self.label_img[y_min:y_max, x_min:x_max]
                try:
                    fg = cv.bitwise_and(roi_img, roi_img, mask=inner_binary)
                except:
                    print(roi_img.shape,inner_binary.shape,[x_min,x_max,y_min,y_max])
                bg = roi_img-fg
                new_fg = np.zeros_like(roi_img)
                new_fg[:,:,1] = inner_binary
                new_fg = new_fg + bg
                self.label_img[y_min:y_max, x_min:x_max] = new_fg

        if jump==False:
            # 绘制标注图上所有毛发
            self.label_img = curve_plot(self.label_img, self.result, self.distinguishValue/self.downsample_ratio,(255, 0, 0),alpha=self.plot_alpha, roi=self.roiInf_corrected)
           # self.qmutex_img_plot.lock()
            self.img_plot[y0:y1, x0:x1] = self.label_img
            #self.qmutex_img_plot.unlock()
            if DEBUG:
                self.plotEndpoints(self.endpoints)
        curve = self.getBinary(update)  # 得到二值图ROI
        curve = curve[y0:y1, x0:x1]
        if len(self.result)>0 and self.handle_index!=-2:
            self.label_img = curve_plot(self.label_img,[self.result[self.handle_index]], 0,(0, 255, 0),(0, 255, 0),alpha=self.plot_alpha,handle_diff=[self.roiInf_corrected[0],self.roiInf_corrected[2]])   # 仅对选中的一根毛发进行绘制
            curve = curve_plot(curve, [self.result[self.handle_index]], 0, (0, 255, 0), (0, 255, 0),
                              alpha=1, handle_diff=[self.roiInf_corrected[0], self.roiInf_corrected[2]], handle_width=2)        # 对二值图进行绘制

        self.setImg2Label(self.label_img,curve)


    def setImg2Label(self,image,curve=None):
        # 绘制标注图
        img = cv.resize(image, self.img_size)
        qimg = QtGui.QImage(img, img.shape[1], img.shape[0],
                            img.shape[1] * 3, QtGui.QImage.Format_RGB888)  # bytesPerLine参数设置为image的width*image.channels
        self.labelImg.setPixmap(QtGui.QPixmap(qimg))

        if curve is not None:
            # 绘制二值图
            curve = cv.resize(curve, self.img_size)
            if len(curve.shape) == 2:
                qimg = QtGui.QImage(curve, curve.shape[1], curve.shape[0],
                                    curve.shape[1] * 1,
                                    QtGui.QImage.Format_Grayscale8)  # bytesPerLine参数设置为image的width*image.channels
            else:
                qimg = QtGui.QImage(curve, curve.shape[1], curve.shape[0],
                                    curve.shape[1] * 3,
                                    QtGui.QImage.Format_RGB888)  # bytesPerLine参数设置为image的width*image.channels
            self.labelImg_curve.setPixmap(QtGui.QPixmap(qimg))


    def updateAll(self,update=False):
        self.imshow(update=update)
        self.updateAnalysis()
        self.saveJson()


    def updateAnalysis(self):
        # 绘制统计信息图
        self.width_count = np.zeros(30, dtype=np.uint16)
        for result_ in self.result:
            width = result_['width']
            if (width > 30):  # 暂时认为不存在超过30宽度的毛发，后面改成自适应数组
                continue
            self.width_count[int(width)] += 1
        inf = analysisInf(self.width_count[:self.distinguishValue])
        self.statusNum.setText(inf['num'])
        self.statusMedian.setText(inf['median'])
        self.statusMean.setText(inf['mean'])
        self.statusStd.setText(inf['std'])
        self.statusMode.setText(inf['mode'])
        inf = analysisInf(self.width_count[self.distinguishValue:], offset=self.distinguishValue)
        self.statusNum_2.setText(inf['num'])
        self.statusMedian_2.setText(inf['median'])
        self.statusMean_2.setText(inf['mean'])
        self.statusStd_2.setText(inf['std'])
        self.statusMode_2.setText(inf['mode'])

        # 显示直方图
        self.plot_widget.clear()
        bg1 = pg.BarGraphItem(x=self.x[:self.distinguishValue], height=self.width_count[:self.distinguishValue],
                              width=1.2, brush=pg.mkBrush(color=(200, 0, 0)))
        bg2 = pg.BarGraphItem(x=self.x[self.distinguishValue:], height=self.width_count[self.distinguishValue:],
                              width=1.2, brush=pg.mkBrush(color=(0, 0, 255)))
        self.plot_widget.addItem(bg1)
        self.plot_widget.addItem(bg2)

    def saveJson(self):
        json_str = json.dumps(self.result, indent=4, cls=NpEncoder)
        with open(self.tmp + '.json', 'w') as json_file:
            json_file.write(json_str)

    def magnetChecked(self, isChecked):
        if (isChecked == 0):
            self.magnet_flag = False
        else:
            self.magnet_flag = True







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
    def binaryChecked_Inner(self,isChecked) :
        if isChecked:
            self.binary_type = BINARY_INNER
            self.getBinary(True)
        self.imshow()

    # def binaryChecked_AutoWithDL(self,isChecked) :
    #     if isChecked:
    #         self.binary_type = BINARY_AUTO_WITH_DL
    #         self.getBinary(True)
    #     self.imshow()

    def innerClear(self):
        try:
            self.inner_stack=[]
            self.imshow()
        except:
            print('clear failed')

    def handleModeChanged_Auto(self,isChecked) :
        if isChecked:
            self.handle_index = -2
            self.innerClear()
            self.handleMode = HANDLE_INNER
    def handleModeChanged_HandleQ(self,isChecked) :
        if isChecked:
            self.handle_index = -2
            self.innerClear()
            self.handleMode = HANDLE_Q
    def handleModeChanged_HandleE(self,isChecked) :
        if isChecked:
            self.handle_index = -2
            self.innerClear()
            self.handleMode = HANDLE_E

    
    def buttonSaveEvent(self):
        try:
            f = open('result.csv', 'w', encoding='utf-8',newline ="")
            csv_writer = csv.writer(f)
            csv_writer.writerow(["index",'sum']+np.linspace(1,24,24).tolist())
            json_files = glob.glob(imgPath + '*.json')
            for json_file in json_files:
                result, result_origin = self.loadJson(json_file)
                width_count = np.zeros(25)
                sum = 0
                for result_ in result:
                    width = result_['width']
                    if(width>24):
                        continue
                    width_count[int(width)] += 1
                    sum += 1

                csv_writer.writerow([os.path.basename(json_file).split('.')[0], sum] + width_count[1:].tolist())
            f.close()

            msg_box = QMessageBox.about(self, '导出结果', '已导出结果至 result.csv')
            #msg_box.exec_()
        except:
            msg_box = QMessageBox(QMessageBox.Warning, '导出结果', '导出失败，请检查csv文件是否被占用')
        msg_box.exec_()
      #  msg_box = QMessageBox(QMessageBox.Warning, '警告', '已导出结果至 result.csv')


    def getBinary(self, update=False):

        def closeopration(binary,ksize=3):
            if ksize==0:
                return binary
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))
            binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
            return binary

        try:
            if self.binary_type==BINARY_NORMAL:
                if update:
                    img_gray = cv.cvtColor(self.image_origin, cv.COLOR_BGR2GRAY)
                    _, self.img_binary = cv.threshold(img_gray, self.binary_threshold_normal, 255, cv.THRESH_BINARY_INV)
                    self.img_binary = closeopration(self.img_binary,ksize=self.binary_close)
                return self.img_binary

            elif self.binary_type==BINARY_AUTO:
                if update:
                    img_gray = cv.cvtColor(self.image_origin, cv.COLOR_BGR2GRAY)
                    self.binary_auto = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,int(11/self.downsample_ratio),self.binary_threshold_auto)
                    self.binary_auto = closeopration(self.binary_auto,ksize=self.binary_close)
                return self.binary_auto

        except:
            # msg_box = QMessageBox(QMessageBox.Warning, '错误', '二值图加载失败，请检查是否有文件缺失')
            # msg_box.exec_()
            self.log('未检测到语义分割图，切换为局部自适应')
            self.binary_type == BINARY_AUTO
            img_gray = cv.cvtColor(self.image_origin, cv.COLOR_BGR2GRAY)
            self.binary_auto = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,
                                                    int(11 / self.downsample_ratio), self.binary_threshold_auto)
            self.binary_auto = closeopration(self.binary_auto, ksize=self.binary_close)
            return self.binary_auto
        # elif self.binary_type==BINARY_AUTO_WITH_DL:     # 要改成：在线程里对整图进行运算，计算好之后直接读取，而不是每次都再运算一次
        #     if update:
        #         img_path = 'data/masks/'+self.tmp.split('/')[-1]+'.png'
        #         dl_output = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY)
        #         _, self.binary_dl = cv.threshold(dl_output, 50, 255, cv.THRESH_BINARY)
        #         self.binary_aoto_with_dl = process.binary_search(self.image_origin,self.binary_dl)
        #     return self.binary_aoto_with_dl






        # elif self.binary_type==BINARY_Cluster:     # 要改成：在线程里对整图进行运算，计算好之后直接读取，而不是每次都再运算一次
        #     if update:
        #         img_gray = cv.cvtColor(self.image_origin, cv.COLOR_BGR2GRAY)
        #         self.binary_cluster = process.binart_cluster(img_gray)
        #     return self.binary_cluster

        return None

    def setResolution(self,scalep):
        def findChildrenWidget(widget, allWidgetList):
            if len(widget.children()) > 0:
                for cwidget in widget.children():
                    allWidgetList.append(cwidget)
                    findChildrenWidget(cwidget, allWidgetList)
            return allWidgetList


        x0 = int(scalep * self.geometry().width())
        y0 = int(scalep * self.geometry().height())
        self.resize(x0, y0)


        self.fontSize =  round(self.fontSize * scalep)
        font = QtGui.QFont()
        font.setFamily("Arial")  # 括号里可以设置成自己想要的其它字体
        font.setPixelSize(self.fontSize)  # 括号里的数字可以设置成自己想要的字体大小
        #font.setPointSize(self.fontSize)  # 括号里的数字可以设置成自己想要的字体大小

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

            try:
                widget.setFont(font)
            except:
                pass
        self.setFixedSize(x0, y0)



        # pyqtgraph柱状图
        self.plot_widget = PlotWidget(self)
        self.plot_widget.setGeometry(QtCore.QRect(int(2000* scalep), int(1430* scalep), int(250* scalep), int(250* scalep)))
        self.width_count = np.zeros(30)
        self.x = np.arange(30)
        y = np.zeros(30)
        bg = pg.BarGraphItem(x=self.x, y=y, height=0, width=0.8* scalep)
        self.plot_widget.addItem(bg)

        #self.graph.addWidget(self.plot_widget)
        for i in range(self.graph.count()):
            self.graph.itemAt(i).widget().deleteLater()

        self.graph.addWidget(self.plot_widget)
        self.distinguishValue = self.sliderDistinguish.value()

    def changeResolution(self,ratio,scalep):
        self.setResolution(ratio / scalep)
        self.scalep = ratio * self.scalepFix
        self.img_size = (self.labelImg.width(), self.labelImg.height())
        self.roi_size = (self.labelImg_roi.width(), self.labelImg_roi.height())
        self.imshow()
        self.imshow_small_picture()

    def resolution70(self):
        self.changeResolution(0.7,self.scalep)

    def resolution80(self):
        self.changeResolution(0.8, self.scalep)

    def resolution90(self):
        self.changeResolution(0.88, self.scalep)

    def resolution100(self):
        self.changeResolution(0.95, self.scalep)




    def distinguishUpdate(self,value):
        self.distinguishValue = int(value)
        self.spinBoxDistinguish.setValue(int(value))
        self.updateAll()

    def thresholdBinaryNormalUpdate(self,value):
        self.binary_threshold_normal = int(value)
        # self.editThreshold.setText(str(value))
        self.imshow(update=True)

    def thresholdBinaryAutoUpdate(self,value):
        self.binary_threshold_auto = int(value)
        # self.editThreshold_2.setText(str(value))

        self.imshow(update=True)

    def binaryCloseUpdate(self,value):
        self.binary_close = int(value)
        self.imshow(update=True)


    # 子线程1：拔毛
    def buttonImpaintEvent(self):
        self.log('拔毛中...')
        self.inpaintThread = InpaintThread(self.result, self.image_origin.copy())  # 步骤2. 主线程连接子线
        self.inpaintThread.mysignal.connect(self.buttonImpaintEventSignal)
        self.inpaintThread.start()  # 步骤3 子线程开始执行run函数
    def buttonImpaintEventSignal(self,img_inpaint):
        self.img_impaint = img_inpaint
        self.imshow_small_picture()
        self.log('拔毛完成!')


    def log(self,text):
        #self.logBrowser.append(time.strftime("[%H:%M:%S]  "+text,time.localtime()))
        self.logBrowser.append(time.strftime("[%H:%M:%S]  ", time.localtime())+text)
        #self.logBrowser.append()
        self.logBrowser.ensureCursorVisible()


    def plotEndpoints(self,endpoints):
        self.img_plot = endpoint_plot(self.img_plot, endpoints, (255, 0, 0),alpha=self.plot_alpha)



    def showMatplotlibBar(self):
        # matplotlib绘制更好看的柱状图。。。。
        plt.figure()
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        num_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        plt.bar(num_list, self.width_count[:20], color="lightseagreen", tick_label=num_list)
        for a, b in zip(num_list, self.width_count[:20]):
            plt.text(a, b + 0.1, b, ha='center', va='bottom')  # 每个柱顶部显示数值
        plt.xlabel('宽度')
        plt.ylabel('数量')
        plt.title(os.path.basename(self.tmp.split('.')[0]))
        plt.show()




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


    def fullImagePlotStart(self):
        self.fullImagePlotThread.update(self.result,self.image_origin,self.distinguishValue,self.plot_alpha)
        self.fullImagePlotThread.signal_img.connect(self.fullImagePlotEventSignal)
        self.fullImagePlotThread.start()


    def fullImagePlotEventSignal(self,img):
        self.img_plot = img
        # x0, x1, y0, y1 = self.roiInf_corrected
        # if self.autoDetectThread is not None and self.autoDetectThread.OVER==True:
        #     print(111111)
        #     self.imshow(jump=True)




    # 列表图片选取
    def itemClick(self):  #列表框单击事件
        self.handle_index = -2
        self.waitForSegmentation = False
        self.binarySaved = False
        self.innerClear()

        self.endpoints = []
        # self.distinguishValue = 0   # 二分类宽度阈值


        self.tmp = imgPath + self.allFiles.currentItem().text()  #图像的绝对路径
        src = cv.imread(str(self.tmp),1)      #读取图像
        self.widthHeightRatio = src.shape[1]/src.shape[0]

        if src.shape[0]<1024:
            self.downsample_ratio = 1
        src = cv.resize(src,(int(src.shape[1]/self.downsample_ratio),int(src.shape[0]/self.downsample_ratio)))



        # 加载标注文件
        self.result, self.result_origin = self.loadJson(self.tmp + '.json')


        src = cv.cvtColor(src, cv.COLOR_BGR2RGB)
        self.image_filter = cv.medianBlur(src, 3)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
        self.image_filter = cv.filter2D(self.image_filter, -1, kernel=kernel)


        #self.image_filter = src





        self.image_origin = src.copy()





        self.img_plot = src.copy()
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

        self.updateAll(update=True)

        self.img_plot = curve_plot(self.image_origin, self.result, self.distinguishValue / self.downsample_ratio,
                                    (255, 0, 0), alpha=self.plot_alpha)
        self.fullImagePlotStart()



    def checkThisHair(self):
        if self.handle_bone == True:
            self.handle_bone = False
            self.result.pop(-1)

            if len(self.result_bone[-1]) >= 2:
                joint_temp = self.result_bone[-1]
               # joint_temp = midUpsample(joint_temp)

                height = process.waist_old(joint_temp,self.getBinary())

                mid = getMidPoint(self.result_bone[-1])
                self.result.append({'joints': self.result_bone[-1], 'width': height, 'mid': mid})


    def generateHairPath(self):
        if self.handle_bone == True:
            self.handle_bone = False
            if len(self.result_bone[-1]) >= 2:
                joint_temp = self.result_bone[-1]

                joint_temp = process.border(joint_temp,self.getBinary())
                def get_roi(joint1,joint2):
                    [x0, y0], [x1, y1] = joint1, joint2
                    x_min = max(min(joint1[0], joint2[0]) - 30, 0)
                    x_max = min(max(joint1[0], joint2[0]) + 30, self.image_origin.shape[1])
                    y_min = max(min(joint1[1], joint2[1]) - 30, 0)
                    y_max = min(max(joint1[1], joint2[1]) + 30, self.image_origin.shape[0])
                    return x0,y0,x1,y1,x_min,x_max,y_min,y_max
                # [x0,y0],[x1,y1] = joint_temp[0],joint_temp[-1]
                # x_min = max(min(joint_temp[0][0], joint_temp[-1][0]) - 30, 0)
                # x_max = min(max(joint_temp[0][0], joint_temp[-1][0]) + 30,self.image_origin.shape[1])
                # y_min = max(min(joint_temp[0][1], joint_temp[-1][1]) - 30,0)
                # y_max = min(max(joint_temp[0][1], joint_temp[-1][1]) + 30,self.image_origin.shape[0])

                joints=[]
                for j in range(len(joint_temp)-1):
                    x0, y0, x1, y1, x_min, x_max, y_min, y_max = get_roi(joint_temp[j], joint_temp[j+1])
                    img_binary = self.getBinary()[y_min:y_max, x_min:x_max]  # 得到二值图ROI
                    joint,length = process.generate_path(img_binary,[y0-y_min,x0-x_min],[y1-y_min,x1-x_min])
                    if len(joint) < 2:
                        continue
                    for joint_ in joint:  # 映射回原图坐标系
                        joint_[0] = joint_[0] + x_min
                        joint_[1] = joint_[1] + y_min
                    joints = joints[:-1]+joint

                if len(joints) < 2:
                    return



                self.result.pop(-1)


                height = process.waist(joints,self.getBinary()/255)
                mid = getMidPoint(joints)

                self.result.append({'joints': joints, 'width': height, 'mid': mid})

           

    # 捕捉键盘事件


    def keyPressEvent(self, QKeyEvent):
        if self.img_loaded==False:
            return

        if QKeyEvent.key() == Qt.Key_Q:  # 回车
            if self.handleMode==HANDLE_Q:
                self.generateHairPath()
            elif self.handleMode==HANDLE_E:
                self.checkThisHair()
            elif self.handleMode==HANDLE_INNER:
                if len(self.inner_stack) <= 0:
                    return
                x_min = 9999
                x_max = 0
                y_min = 9999
                y_max = 0
                for [x0, x1, y0, y1], _ in self.inner_stack:
                    x_min = min(x_min, x0)
                    x_max = max(x_max, x1)
                    y_min = min(y_min, y0)
                    y_max = max(y_max, y1)
                inner_binary_merged = np.zeros((y_max - y_min, x_max - x_min), np.uint8)
                for [x0, x1, y0, y1], roi_binary in self.inner_stack:
                    inner_binary_merged[(y0 - y_min):(y1 - y_min), (x0 - x_min):(x1 - x_min)] += roi_binary
                inner_binary_merged = (inner_binary_merged > 0).astype(np.uint8)

                _, hair_pairs = process.autoSkeletonExtraction(inner_binary_merged,max_num_hairs=3)

                temp_result = []
                for joints in hair_pairs:
                    height = process.waist(joints, inner_binary_merged)
                    for joint in joints:
                        joint[0] = joint[0] + x_min
                        joint[1] = joint[1] + y_min
                    mid = getMidPoint(joints)
                    temp_result.append({'joints': joints, 'width': height, 'mid': mid})
                self.result += temp_result

                self.inner_stack = []

            self.updateAll()
            return

        #参数1  控件
        if(len(self.result)<1):
            return
        if QKeyEvent.key()== Qt.Key_E:  # 删除
            if self.handleMode == HANDLE_INNER:
                self.innerClear()
                self.imshow()


            if self.handle_index!=-2:
                self.handle_bone = False
                self.result.pop(self.handle_index)


                if len(self.result)==0:
                    self.handle_index = -2
                else:
                    self.handle_index = -1
                self.updateAll()

        if (len(self.result) < 1):
            return

        joints = self.result[self.handle_index]['joints']
        width = self.result[self.handle_index]['width']
        mid = self.result[self.handle_index]['mid']

        #rect=self.result[self.handle_index]['rect']
       # [[x,y],[w,h],angle] = rect


        if QKeyEvent.key() == Qt.Key_A:  # 左移
            for joint in joints:
                joint[0] -= 1
            mid = getMidPoint(joints)
            self.result.pop(self.handle_index)
            self.result.append({'joints':joints, 'width':width,'mid':mid})

            self.handle_index = -1      # 修正后，对应目标的序号必定变为-1
            self.imshow()


        elif QKeyEvent.key()== Qt.Key_D:  # 右移
            for joint in joints:
                joint[0] += 1
            mid = getMidPoint(joints)
            self.result.pop(self.handle_index)
            self.result.append({'joints':joints, 'width':width,'mid':mid})

            self.handle_index = -1  
            self.imshow()


        elif QKeyEvent.key()== Qt.Key_S:  # 下移
            for joint in joints:
                joint[1] += 1
            mid = getMidPoint(joints)
            self.result.pop(self.handle_index)

            self.result.append({'joints':joints, 'width':width,'mid':mid})

            self.handle_index = -1  
            self.imshow()

        elif QKeyEvent.key()== Qt.Key_W:  # 上移
            for joint in joints:
                joint[1] -= 1
            mid = getMidPoint(joints)
            self.result.pop(self.handle_index)
            self.result.append({'joints':joints, 'width':width,'mid':mid})
            self.handle_index = -1  
            self.imshow()


        elif QKeyEvent.key()== Qt.Key_Up:  # 变宽
            width += 1
            self.result.pop(self.handle_index)
            self.result.append({'joints':joints, 'width':width,'mid':mid})
            self.handle_index = -1  
            self.updateAll()

        elif QKeyEvent.key()== Qt.Key_Down:   # 变窄
            width -= 1
            self.result.pop(self.handle_index)
            self.result.append({'joints':joints,  'width':width,'mid':mid})
            self.handle_index = -1  
            self.updateAll()




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
                if data['width']<=1 or isnan(data['mid'][0]):
                    continue

                # print(data['mid'][0])
                # print())
                # a = data['mid'][0]
                # b = nan in data['mid']


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
    app.setFont(QFont("微软雅黑", 9))
    app.setWindowIcon(QIcon("icon.ico"))
    stylesheet = qdarkstyle.load_stylesheet_pyqt5()
    app.setStyleSheet(stylesheet)
    MainWindow = Mark()
    MainWindow.show()
    app.installEventFilter(MainWindow)
    sys.exit(app.exec_())