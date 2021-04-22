from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5 import uic
from PIL import Image
import cv2
import numpy as np

class AuthorWindow(QMainWindow):
    def __init__(self):
        super(AuthorWindow, self).__init__()
        uic.loadUi('assets/author.ui', self)
        self.showImage()
    def showImage(self):
        imgs = [Image.open('assets/img1.jpg'),
                Image.open('assets/img2.jpg'),
                Image.open('assets/img3.jpg'),
                Image.open('assets/img4.jpg'),
                Image.open('assets/img5.jpg'),
                Image.open('assets/img6.jpg'),
                Image.open('assets/img7.jpg'),
                Image.open('assets/img8.jpg')]
        labels = [self.img1,self.img2,self.img3,self.img4,self.img5,self.img6,self.img7,self.img8]

        for src,label in zip(imgs,labels):
            img = cv2.resize(np.array(src), (128, 174))
            qimg = QtGui.QImage(img, img.shape[1], img.shape[0],
                                img.shape[1] * 3,
                                QtGui.QImage.Format_RGB888)  # bytesPerLine参数设置为image的width*image.channels
            label.setPixmap(QtGui.QPixmap(qimg))
