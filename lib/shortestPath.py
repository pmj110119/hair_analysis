"""
    by wjj 2021/2/27
    modified by cdc  2021/3/1
"""       

import cv2
import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import heapq
from lib.smoothing import skeleton_smoothing

from numba import jit

@jit(nopython = True)
def dijkstra(Img, D, Start, Endlist):
    
    rows, cols = D.shape[:2]
    stx, sty = Start
    targets = set()
    for enx, eny in Endlist:
        if ( (enx, eny) != (stx, sty) ):
            targets.add((enx, eny))
    
    # Dijkstra最短路径算法
    ### 初始化
    dist = np.ones((rows, cols)) * np.inf
    dist[stx, sty] = D[stx, sty]
    lastx = np.ones((rows, cols), dtype = np.int32) * (-1)
    lasty = np.ones((rows, cols), dtype = np.int32) * (-1)
    heap = [(0., (stx, sty))]
    heapq.heapify(heap)
    ### 迭代

    connected_target = 0
    while (True):
            
        nowx, nowy, nowdist = -1, -1, np.inf
        while len(heap):
            tmpdist, (tmpx, tmpy) = heapq.heappop(heap)
            if  (tmpdist > dist[tmpx, tmpy]): # node `now` has been visited before
                continue
            else:
                nowx, nowy, nowdist = tmpx, tmpy, tmpdist
                break
        if (nowx == -1) :
            break
        dist[nowx, nowy] = nowdist
        if ( (nowx, nowy) in targets):
            targets.remove((nowx, nowy))
            connected_target += 1

        if (len(targets) == 0) or (nowdist > 10000.) : # there is no need to explore further
            break

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if (dx != 0) or (dy != 0):
                    nextx = np.int32(nowx + dx)
                    nexty = np.int32(nowy + dy)
                    if nextx >= 0 and nextx < rows and nexty >= 0 and nexty < cols:
                        nextdist = nowdist + D[nextx, nexty]
                        if (nextdist < dist[nextx, nexty]):
                            dist[nextx, nexty] = nextdist
                            lastx[nextx, nexty] = nowx
                            lasty[nextx, nexty] = nowy
                            heapq.heappush(heap, (nextdist, (nextx, nexty)))
    
    result = []
    
    for enx, eny in Endlist:
    
        # 沿着每个点被访问时的父亲，找出方案
        shortest_path = [(enx, eny)]
        tmpx, tmpy = enx, eny
        while (lastx[tmpx, tmpy] > -1):
            tmpx, tmpy = (lastx[tmpx, tmpy], lasty[tmpx, tmpy])
            shortest_path.append((tmpx, tmpy))
        shortest_path.reverse()
        
        result.append( (shortest_path, dist[enx, eny]) )

    return result

def getShortestPath(Img, Start, End):
    
    Img = Img.astype(np.uint8)

    D = ndimage.distance_transform_edt(Img == 1)
    D = np.max(D) - D + 1
    D[Img != 1] = 10000.
    
    x0, y0 = map(np.int32, Start)
    x1, y1 = map(np.int32, End)
    
    shortest_path, dist = dijkstra(Img, D, (x0, y0), np.array([(x1, y1)]) )[0]
    shortest_path = skeleton_smoothing(Img, shortest_path)
    return shortest_path, dist
 
if __name__ == "__main__":
    import time
    
    img = cv2.imread('binary.jpg', cv2.IMREAD_GRAYSCALE)
    img = (img > 127).astype(np.float32)
    img = img[:400, :400]

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
                StartPoint = (ypos,xpos)
                print('1:',StartPoint)
            elif ClickNum==2:
                EndPoint = (ypos,xpos)
                print('2:',EndPoint)
                print('Calculating....')
                
                t0 = time.time()
                shortestpath, dis = getShortestPath(img,StartPoint,EndPoint)
                print("Time: ", time.time() - t0, " s")
                print(dis, shortestpath.tolist())

                if (len(shortestpath)):
                    shortestpath = np.array(shortestpath)
                    img[shortestpath[:, 0], shortestpath[:, 1]] = 0.5
                    cv2.imshow('image',img)
                
                ClickNum = 0

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', action)
    ClickNum = 0
    StartPoint = [0,0]
    EndPoint = [0,0]

    cv2.imshow('image',img)
    while True:
        key = cv2.waitKey(1)
        if key==27: #按Esc
            cv2.destroyAllWindows()
            break
        if ClickNum>=2:
            continue
            
    cv2.destroyAllWindows()
                