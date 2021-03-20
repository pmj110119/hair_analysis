import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage
from lib.shortestPath import dijkstra

def evaluate_path(path):
    """
        打分估计`path`的形状是否像一根毛发。
    """
    path = np.array(path, dtype = np.float32)
    N = len(path)
    if (N <= 12):
        return 0
    dirs = path[10:, :] - path[:-10, :]
    
    x1, y1, x2, y2 = dirs[:-1, 0], dirs[:-1, 1], dirs[1:, 0], dirs[1:, 1]
    angle_diff = np.arccos((x1 * x2 + y1 * y2) / np.sqrt((x1*x1 + y1*y1) * (x2*x2 + y2*y2)))
    
    return 1./(np.mean(angle_diff**2) + 1e-6)

def skeletonExtraction(img_binary, endpoints):
    """
        输入二值图和检测到的端点，返回找到的所有毛发。
    """
    img_binary = img_binary.astype(np.uint8)
    D = ndimage.distance_transform_edt(img_binary == 1)
    D = np.max(D) - D + 1
    D[img_binary != 1] = 10000.
    
    if (not isinstance(endpoints, np.ndarray)):
        endpoints = np.array(np.endpoints)
    endpoints = endpoints.astype(np.int32)
    
    N = len(endpoints)
    edge = []
    all2all = []
    for i in range(N):
        i2all = dijkstra(img_binary, D, endpoints[i], endpoints)
        all2all.append(i2all)
        for j in range(i):
            if (j != i) and (i2all[j][1] < np.inf):
                edge.append( (i, j, evaluate_path(i2all[j][0])) ) 
    
    # 贪心法近似估计一般图最大权匹配
    paths = []
    edge.sort(key = lambda ele : ele[2], reverse = True)
    vis = set()
    for u, v, weight in edge:
        if (u in vis) or (v in vis) or (np.isnan(weight)):
            continue
        vis.add(u)
        vis.add(v)
        paths.append( all2all[u][v][0] )
    
    return paths

if (__name__ == "__main__"):
    
    from endpointDetection import endpointDetection
    
    img = cv2.imread('binary.jpg', cv2.IMREAD_GRAYSCALE)
    binary = (img > 127).astype('uint8')
    #binary = binary[:400, :400]
    
    print("Finding endpoints...")
    endpoints = endpointDetection(binary)
    print("Extracting skeletons...")
    paths = skeletonExtraction(binary, endpoints)

    plt.imshow(binary, cmap = plt.cm.gray)
    for path in paths:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], markersize = 1, c = 'red', marker = '.', linewidth = 0)
    plt.savefig("paths.pdf", dpi = 1200)
    
    #savemat("end_points.mat", {"end_points" : ans})
