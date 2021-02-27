from sklearn import cluster
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")


def myAggCluster(img,n_clusters):
    img = 255.0 - img
    D_x, D_y = img.shape[:2]
    value0 = np.zeros((D_x * D_y, 1))
    num = -1
    for yi in range(D_y):
        for xi in range(D_x):
            num += 1
            value0[num] = img[xi][yi]

    clust = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    label0 = clust.fit_predict(value0)

    # 根据均值判断
    mean_value_label = np.zeros((3, 1))
    for i_cluster in range(n_clusters):
        s = [value0[i] for i in range(len(label0)) if (label0[i] == i_cluster)]
        mean_value_label[i_cluster] = np.mean(np.array(s))
    label_choose = np.argmax(mean_value_label)

    LabelMatrix = np.zeros((D_x, D_y),dtype=np.uint8)
    num = -1
    for yi in range(D_y):
        for xi in range(D_x):
            num += 1
            if label0[num] == label_choose:
                LabelMatrix[xi][yi] = 1
    return LabelMatrix*255

if __name__ == "__main__":
    image_dir = "./test.png"
    img = cv2.imread(image_dir,cv2.IMREAD_GRAYSCALE)

    n_clusters = 3
    LabelMatrix = myAggCluster(img, n_clusters)
    #cv2.namedWindow('Label')
    cv2.imwrite('Label.png', LabelMatrix)
    cv2.waitKey(0)