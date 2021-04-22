import numpy as np
from lib.skeletonExtraction import skeletonExtraction
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt

def filter(endpoints, x1, y1, x2, y2):
    if (endpoints is None):
        return None
    tmp = (endpoints[:, 0] >= x1) & (endpoints[:, 0] < x2) & (endpoints[:, 1] >= y1) & (endpoints[:, 1] < y2)
    return endpoints[tmp, :] - np.array([[x1, y1]], dtype = endpoints.dtype)

def autoSkeletonExtraction(binary, display = False, endpoints = None, **kwargs):

    binary = binary.astype(np.uint8)
    n, m = binary.shape

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 
                                                                     connectivity = 8)
    ans = []
    for i in tqdm(range(1, num_labels), ncols = 50) if display else range(1, num_labels):
        y, x, width, height, _ = stats[i]
        x_min = max(x - 30, 0)
        y_min = max(y - 30, 0)
        x_max = min(x + height + 30, n)
        y_max = min(y + width + 30, m)
        mask = (labels[x_min:x_max, y_min:y_max] == i).astype(np.uint8)
        # mask = (labels==i)[x_min:x_max, y_min:y_max].astype(np.uint8)  
        # may be a little bit slower
        paths = skeletonExtraction(mask, endpoints = filter(endpoints, x_min, y_min, x_max, y_max), **kwargs) 
        for path in paths:
            path[:, 0] += x_min
            path[:, 1] += y_min
        ans = ans + paths

    return ans

if (__name__ == "__main__"):
    
    filename = "test/114BGAYJ.jpg"
    #filename = 'test/binary.jpg'
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    binary = (img > 127).astype(np.uint8)

    endpoints = np.load("test/114BGAYJ.jpg.npy").astype(np.int32)
    endpoints_new = np.zeros(endpoints.shape, dtype = np.int32)
    endpoints_new[:, 0], endpoints_new[:, 1] = endpoints[:, 1], endpoints[:, 0] 
    endpoints = endpoints_new

    print("Automatically extracting all the skeletons...")
    
    paths = autoSkeletonExtraction(binary, display = True, max_num_hairs = 3, endpoints = endpoints, refind = True)

    plt.imshow(binary, cmap = plt.cm.gray)
    colors = ["red", "green", "blue", "yellow", "cyan", "deeppink", "purple"]
    color_id = 0
    for path in paths:
        color_id = (color_id + 1) % len(colors)
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], markersize = 1, c = colors[color_id], marker = '.', linewidth = 0)
        #plt.text(path[0, 1], path[0, 0], str(info), fontsize = 2, color = 'blue')
    plt.savefig(filename + ".paths_all.jpg", dpi = 1200)
                                             