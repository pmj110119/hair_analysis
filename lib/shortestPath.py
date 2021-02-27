"""
    by wjj 2021/2/27
"""       

import cv2
import cv2
import numpy as np
from scipy import ndimage
from lib.myGraph import *
import matplotlib.pyplot as plt

def getBinImage(image_dir,thr):
    image0 = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
    image = np.zeros(image0.shape)
    image[image0>thr]=1
    #cv2.imwrite('zzzz.png',image*255)
    return image
    # img = plt.imshow(image)
    # img.set_cmap('gray')
    # plt.show()

def sub2ind(array_shape, cols, rows):
    return rows*array_shape[0] + cols

def ind2sub(array_shape, ind):
    rows = (ind // array_shape[0])
    cols = (ind % array_shape[0])
    return (cols,rows)

def my_im2graph(Img,N_conn):
    I = Img
    D0 = ndimage.distance_transform_edt(I == 1)
    D = np.max(D0) - D0 + 1
    D_x, D_y = D.shape[:2]
    NodeNum = D_x*D_y
    s = []
    t = []
    w = []
    NUM = -1
    for yind in range(D_y):
        for xind in range(D_x):
            NUM = NUM + 1
            stmp = []
            ttmp = []
            wtmp = []
            for yi in [-1,0,1]:
                for xi in [-1,0,1]:
                    if (xi==0 and yi==0)==False:
                        xindx = xind + xi
                        yindx = yind + yi
                        if xindx>=0 and xindx<D_x and yindx>=0 and yindx<D_y and I[xind,yind]==1 and I[xindx,yindx]==1:
                            stmp += [NUM]
                            ttmp += [sub2ind(D.shape,xindx,yindx)]
                            wtmp += [(D[xind,yind]+D[xindx,yindx])/2]
            s += stmp
            t += ttmp
            w += wtmp

    G = Graph()
    gmat = {}
    for svalue,tvalue,wvalue in zip(s,t,w):
        if svalue not in gmat.keys():
            gmat[svalue] = {}
        gmat[svalue][tvalue] = wvalue
    for key_i in gmat.keys():
        for key_j in gmat[key_i].keys():
            e = Edge(key_i, key_j, gmat[key_i][key_j])
            G.add_edge(e)
    return G


def getShortestPath(Img,StartPoint,EndPoint):
    try:
        StartInd = sub2ind(Img.shape, StartPoint[0], StartPoint[1])
        EndInd = sub2ind(Img.shape,EndPoint[0],EndPoint[1])
        G = my_im2graph(Img,8)
        path,pathdist = G.find_shortest_path(StartInd, EndInd)
        shortestpath = []
        for pathind in path:
            shortestpath += [list(ind2sub(Img.shape,pathind))]
        return shortestpath,pathdist
    except:
        return [],0
if __name__ == "__main__":
    image_dir = "./test.png"
    Binthr = 200
    img = getBinImage(image_dir,Binthr)
    # StartPoint = [78,9]
    # EndPoint = [4,165]
    # shortestpath,pathdist = DijkstraShortestPath(img, StartPoint, EndPoint)
    # print(shortestpath)

    # 鼠标双击的回调函数
    def action(event, x_click, y_click, flags, param):
        global Img,ClickNum,StartPoint,EndPoint
        if event == cv2.EVENT_LBUTTONDOWN:
            xpos = x_click #Width
            ypos = y_click #Height
            ClickNum += 1
            if ClickNum==1:
                StartPoint = [ypos,xpos]
                print('1:',StartPoint)
            elif ClickNum==2:
                EndPoint = [ypos,xpos]
                print('2:',EndPoint)
                cv2.destroyAllWindows()
                print('Calculating....')
                shortestpath,pathdist = getShortestPath(img,StartPoint,EndPoint)
                print(shortestpath)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', action)
    ClickNum = 0
    StartPoint = [0,0]
    EndPoint = [0,0]

    while True:
        cv2.imshow('image',img)
        key = cv2.waitKey(1)
        if key==27: #按Esc
            cv2.destroyAllWindows()
            break
        if ClickNum>=2:
            break


