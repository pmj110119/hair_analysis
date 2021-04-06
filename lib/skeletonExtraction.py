import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage
from lib.shortestPath import dijkstra
from lib.endpointDetection import endpointDetection
from lib.widthEstimation import waist
from lib.intersectionDetection import intersectionDetection, is_intersection
from lib.smoothing import skeleton_smoothing
from lib.pathEvaluation import evaluate_path

from time import time

def remove_hair(binary, path, width = 10):
    """
        输入二值图和一根毛发的骨架，把对应的毛发拔掉。
    """
    is_inter = is_intersection(binary, path)
    mask = np.zeros(binary.shape, dtype = np.uint8)
    
    widths, corrected_path = waist(path, binary)
    N = len(path)
    for i in range(N):
        x, y = corrected_path[i]
        if (widths[i] >= 15) or (is_inter[i]):
            continue
        x, y = corrected_path[i]
        w = np.min(widths[max(i-10, 0) : min(i+10, N)])
        cv2.circle(mask, (y, x),
                    round(w / 2) + 3, 
                    color = 1, 
                    thickness = cv2.FILLED
                    )
    mask *= binary
    
    # cv2.imshow("mask", mask * 255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    dst = cv2.inpaint((binary*255).astype(np.uint8), (mask*255).astype(np.uint8), width, cv2.INPAINT_NS)

    # cv2.imshow("dst", dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return (dst > 50).astype(np.uint8)

def skeletonExtraction_single(img_binary, endpoints = None, refind = False, thres = 0.05, min_length = 30):
    """
        输入二值图，返回一根毛发。
    """

    if (endpoints is None):
        refind = True
    
    # 端点检测
    if (endpoints is None):
        endpoints = endpointDetection(img_binary)
    else:
        endpoints = endpointDetection(img_binary, endpoints, refind = refind)
    if (not isinstance(endpoints, np.ndarray)):
        endpoints = np.array(endpoints)
    endpoints = endpoints.astype(np.int32)  
    N = len(endpoints)
    if (N < 2):
        return None, np.inf
 
    # 利用距离变换，设置每个点的权值
    D = ndimage.distance_transform_edt(img_binary == 1)
    D = np.max(D) - D + 1
    D[img_binary != 1] = 10000.

    # 求解两两端点之间的最短路径
    path, score = None, np.inf
    for i in range(N):
        i2all = dijkstra(img_binary, D, endpoints[i], endpoints)
        for j in range(i):
            dis = i2all[j][1]
            if (j != i) and (dis < np.inf):
                cur_path = np.array(i2all[j][0], dtype = np.int32)
                if (len(cur_path) <= min_length): ##### 长度过短
                    continue
                cur_score = evaluate_path(cur_path)
                if (cur_score > thres): ##### 得分超过某个阈值（越小路径越平滑）
                    continue
                if (cur_score < score):
                    path, score = cur_path, cur_score
                                    
    return path, score

def skeletonExtraction(img_binary, endpoints = None, debug = False, single_hair_mode = False, refind = False, 
                       max_num_hairs = 10, 
                       min_length = 30):
    """
        输入二值图，返回找到的所有毛发。
        方法：逐根查找。找长得最像毛发的一条路径，将其抹去，反复迭代。
    """

    img_binary = img_binary.astype(np.uint8)

    # if (debug):
    #     t0 = time()

    # intersection = intersectionDetection(binary)
    
    # if (debug):
    #     print("Find intersection: ", time() - t0, "s")

    paths = []
    i = 0
    while i < max_num_hairs:
        
        i += 1
        if (debug):
            print("Extracting the %d-th hair...." % i)
            t0 = time()

        path, score = skeletonExtraction_single(img_binary, endpoints, refind = refind, min_length = min_length)
        if (path is None):
            break

        if (debug):
            print("Find hair: ", time() - t0, "s")
            t0 = time()

        img_binary_new = remove_hair(img_binary, path)
        path = skeleton_smoothing(img_binary, path)
        img_binary = img_binary_new
        if (len(path) <= min_length):
            continue

        if (debug):
            print("Remove hair:", time() - t0, "s")
            paths.append( (path, (score, i)) )
            plt.imshow(img_binary, cmap = plt.cm.gray)
            for path, info in paths:
                path = np.array(path)
                plt.plot(path[:, 1], path[:, 0], markersize = 1, c = 'red', marker = '.', linewidth = 0)
            plt.savefig("debug/paths_phase_%d.jpg" % i, dpi = 600)
        else:
            paths.append( path ) 
        
        if (single_hair_mode):
            break
    
    return paths

# 调用：
# (1) paths = skeletonExtraction(binary) 每次迭代，用传统方法重找端点(比较慢)
# 或，如果已经用别的方法找好了端点endpoints (N * 2 array)
# (2) paths = skeletonExtraction(binary, endpoints) 每次都用endpoints这些端点来配对，不重新找
# 或，即使已经找好了端点endpoints，但这只是推荐的端点，希望每次迭代都结合传统找端点方法的结果(比较慢)
# (3) paths = skeletonExtraction(binary, endpoints, refind = True)

# 指定最多提取的毛发数
# paths = skeletonExtraction(..., max_num_hairs = 3)

if (__name__ == "__main__"):
    
    img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
    binary = (img > 127).astype(np.uint8)
    #binary = binary[:200, :200]

    # endpoints = np.load("endpoints.npy")
    # binary = np.load("img_binary.npy")

    print("Extracting skeletons...")
    
    paths = skeletonExtraction(binary, refind = True, debug = True, max_num_hairs = 20)

    plt.imshow(binary, cmap = plt.cm.gray)
    for path, info in paths:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], markersize = 1, c = 'red', marker = '.', linewidth = 0)
        plt.text(path[0, 1], path[0, 0], str(info), fontsize = 2, color = 'blue')
    plt.savefig("paths.pdf", dpi = 1200)
    
    # paths = skeletonExtraction(binary, refind = True, debug = False, max_num_hairs = 20)

    # plt.imshow(binary, cmap = plt.cm.gray)
    # for path in paths:
    #     path = np.array(path)
    #     plt.plot(path[:, 1], path[:, 0], markersize = 1, c = 'red', marker = '.', linewidth = 0)
    #     #plt.text(path[0, 1], path[0, 0], str(info), fontsize = 2, color = 'blue')
    # plt.savefig("paths.pdf", dpi = 1200)
    
    #savemat("end_points.mat", {"end_points" : ans})
