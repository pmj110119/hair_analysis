"""
    by wjj 2021/2/27
    modified by cdc  2021/3/1
"""       

import cv2
import cv2
import numpy as np
from scipy import ndimage
# from lib.myGraph import *
import matplotlib.pyplot as plt
import heapq

from numba import jit

# def sub2ind(array_shape, cols, rows):
#     return rows*array_shape[0] + cols

# def ind2sub(array_shape, ind):
#     rows = (ind // array_shape[0])
#     cols = (ind % array_shape[0])
#     return (cols,rows)

# def my_im2graph(Img,N_conn):
#     I = Img
#     D0 = ndimage.distance_transform_edt(I == 1)
#     D = np.max(D0) - D0 + 1
#     D_x, D_y = D.shape[:2]
#     NodeNum = D_x*D_y
#     s = []
#     t = []
#     w = []
#     NUM = -1
#     for yind in range(D_y):
#         for xind in range(D_x):
#             NUM = NUM + 1
#             stmp = []
#             ttmp = []
#             wtmp = []
#             for yi in [-1,0,1]:
#                 for xi in [-1,0,1]:
#                     if (xi==0 and yi==0)==False:
#                         xindx = xind + xi
#                         yindx = yind + yi
#                         if xindx>=0 and xindx<D_x and yindx>=0 and yindx<D_y and I[xind,yind]==1 and I[xindx,yindx]==1:
#                             stmp += [NUM]
#                             ttmp += [sub2ind(D.shape,xindx,yindx)]
#                             wtmp += [(D[xind,yind]+D[xindx,yindx])/2]
#             s += stmp
#             t += ttmp
#             w += wtmp

#     G = Graph()
#     gmat = {}
#     for svalue,tvalue,wvalue in zip(s,t,w):
#         if svalue not in gmat.keys():
#             gmat[svalue] = {}
#         gmat[svalue][tvalue] = wvalue
#     for key_i in gmat.keys():
#         for key_j in gmat[key_i].keys():
#             e = Edge(key_i, key_j, gmat[key_i][key_j])
#             G.add_edge(e)
#     return G
# def getShortestPath(Img,StartPoint,EndPoint):
#     #try:
# #         StartInd = sub2ind(Img.shape, StartPoint[0], StartPoint[1])
# #         EndInd = sub2ind(Img.shape,EndPoint[0],EndPoint[1])
# #        G = my_im2graph(Img,8)
# #         path,pathdist = G.find_shortest_path(StartInd, EndInd)
# #         shortestpath = []
# #         for pathind in path:
# #             shortestpath += [list(ind2sub(Img.shape,pathind))]
#         Start = (StartPoint[0], StartPoint[1])
#         End = (EndPoint[0], EndPoint[1])
              
#         shortestpath,pathdist = find_shortest_path(Img, Start, End)
#         assert(pathdist is not None)
#         return shortestpath,pathdist
#     #except:
#     #    return [],0

@jit(nopython = True)
def dijkstra(Img, D, Start, End):
    
    if (Img[Start] != 1) or (Img[End] != 1):
        return [End], np.inf
    
    rows, cols = D.shape[:2]
    nodes = []
    idx = np.zeros((rows, cols), dtype = np.int32)
    
    # 给图上空白的格子标号
    N = 0
    for i in range(rows):
        for j in range(cols):
            if (Img[i, j] == 1) :
                idx[i, j] = N
                nodes.append( (np.int32(i), np.int32(j)) )
                N += 1
    start, end = idx[ Start ], idx[ End ]
    
    # Dijkstra最短路径算法
    ### 初始化
    dist = np.ones(N) * np.inf
    dist[start] = 0.
    last = np.ones(N, dtype = np.int32) * (-1)
    heap = [(0., start)]
    heapq.heapify(heap)
    
    ### 迭代
    for _ in range(N-1):
            
        now, nowdist = -1, np.inf
        while len(heap):
            tmpdist, tmp = heapq.heappop(heap)
            if  (tmpdist > dist[tmp]): # node `now` has been visited before
                continue
            else:
                now, nowdist = tmp, tmpdist
                break
        if (now == -1) :
            break
        dist[now] = nowdist
        if (now == end) : # there is no need to explore further
            break

        nowx, nowy = nodes[now]    
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if (dx != 0) or (dy != 0):
                    nextx = nowx + dx
                    nexty = nowy + dy
                    if nextx >= 0 and nextx < rows and nexty >= 0 and nexty < cols and Img[nextx,nexty]==1:
                        nextid = idx[nextx, nexty]
                        nextdist = nowdist + D[nextx, nexty]
                        if (nextdist < dist[nextid]):
                            dist[nextid] = nextdist
                            last[nextid] = now
                            heapq.heappush(heap, (nextdist, nextid))
    
    # 沿着每个点被访问时的父亲，找出方案
    shortest_path = [nodes[end]]
    tmp = end
    while (last[tmp] > -1):
        tmp = last[tmp]
        shortest_path.append(nodes[tmp])
    shortest_path.reverse()
    
    return shortest_path, dist[end] + (D[Start]-D[End])/2 

def getShortestPath(Img, Start, End):
    
    D = ndimage.distance_transform_edt(Img == 1)
    D = np.max(D) - D + 1
    
    x0, y0 = map(np.int32, Start)
    x1, y1 = map(np.int32, End)
    
    return dijkstra(Img, D, (x0, y0), (x1, y1))

def getBinImage(image_dir,thr):
    image0 = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
    image = np.zeros(image0.shape)
    image[image0<thr]=1
    #cv2.imwrite('zzzz.png',image*255)
    return image
    # img = plt.imshow(image)
    # img.set_cmap('gray')
    # plt.show()
    
if __name__ == "__main__":
    import time
    
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
                StartPoint = (ypos,xpos)
                print('1:',StartPoint)
            elif ClickNum==2:
                EndPoint = (ypos,xpos)
                print('2:',EndPoint)
                print('Calculating....')
                
                t0 = time.time()
                shortestpath,pathdist = getShortestPath(img,StartPoint,EndPoint)
                print("Time: ", time.time() - t0, " s")
                print(shortestpath)
                
                if (len(shortestpath)):
                    shortestpath = np.array(shortestpath)
                    img[shortestpath[:, 0], shortestpath[:, 1]] = 0.5;
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
                