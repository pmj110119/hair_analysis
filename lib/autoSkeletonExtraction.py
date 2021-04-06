import numpy as np
from skeletonExtraction import skeletonExtraction
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt

def autoSkeletonExtraction(binary, display = False, **kwargs):

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
        paths = skeletonExtraction(mask, **kwargs) 
        for path in paths:
            path[:, 0] += x_min
            path[:, 1] += y_min
        ans = ans + paths

    return ans

if (__name__ == "__main__"):
    
    filename = "binary.jpg"
    #filename = '124DDBSY.jpg.inner.png'
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    binary = (img > 127).astype(np.uint8)
    #binary = binary[:200, :200]

    # endpoints = np.load("endpoints.npy")
    # binary = np.load("img_binary.npy")

    print("Automatically extracting all the skeletons...")
    
    paths = autoSkeletonExtraction(binary, display = True, max_num_hairs = 3)

    plt.imshow(binary, cmap = plt.cm.gray)
    colors = ["red", "green", "blue", "yellow", "cyan", "deeppink", "purple"]
    color_id = 0
    for path in paths:
        color_id = (color_id + 1) % 7
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], markersize = 1, c = colors[color_id], marker = '.', linewidth = 0)
        #plt.text(path[0, 1], path[0, 0], str(info), fontsize = 2, color = 'blue')
    plt.savefig(filename + ".paths_all.jpg", dpi = 1200)
                                             