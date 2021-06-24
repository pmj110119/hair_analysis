
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

from main import *

DEBUG = False







#图像标记类
class Mark_handle(Mark):
    def __init__(self):
        super(Mark_handle, self).__init__()
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


        self.img_area = float(self.statusArea.text())
        self.pixel_width = 1e4 * sqrt(self.img_area/2048/2560)  # 以um为单位

        self.initUI()










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

        return None




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