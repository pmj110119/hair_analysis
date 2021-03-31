import numpy as np
import cv2
from math import *

def waist(joints, img_binary, r = 15):
    n, m = img_binary.shape

    tmp = np.arange(1, r + 1)[np.newaxis, :]
    degree = (np.arange(0, 360) / 360. * 2 * np.pi)[:, np.newaxis]
    dx = np.floor(tmp * np.cos(degree)).astype(np.int32)
    dx_ = np.ceil(tmp * np.cos(degree)).astype(np.int32)
    dy = np.floor(tmp * np.sin(degree)).astype(np.int32)
    dy_ = np.ceil(tmp * np.sin(degree)).astype(np.int32)
    del tmp

    def calculate_width(binary, x, y, angle_idx):
        
        xx, yy, xx_, yy_ = x + dx[angle_idx], y + dy[angle_idx], x + dx_[angle_idx], y + dy_[angle_idx]
        value1 = np.max((binary[xx, yy], binary[xx, yy_],
                         binary[xx_, yy], binary[xx_, yy_]), axis=0)
        angle_idx = angle_idx + 180
        xx, yy, xx_, yy_ = x + dx[angle_idx], y + dy[angle_idx], x + dx_[angle_idx], y + dy_[angle_idx]
        value2 = np.max((binary[xx, yy], binary[xx, yy_],
                         binary[xx_, yy], binary[xx_, yy_]), axis=0)
        w1, w2 = np.argmax(value1 == 0), np.argmax(value2 == 0)
        w = w1 + w2
        dr = w1 - w/2
        return w, (round(x + dr * np.cos(angle)), round(y + dr * np.sin(angle)))

    shifted = np.zeros((n + 2 * r + 1, m + 2 * r + 1), dtype=np.uint8)
    shifted[r:r + n, r:r + m] = img_binary

    N = len(joints)
    waist_array = np.zeros(N, dtype = np.int32)
    corrected = np.zeros( (N, 2), dtype = np.int32)
    for i in range(0,N):
        x, y = joints[i]
        dir = joints[min(i + 10, N-1)] - joints[max(i - 10, 0)]
        angle = (int(np.arctan2(dir[1], dir[0]) / np.pi * 180.) + 360 + 90) % 180
        waist_array[i], corrected[i] = calculate_width(shifted, x + r, y + r, angle)
    return waist_array, corrected - r
    #print(waist_array)
    #waist_median = round(findNearest(waist_array, np.median(waist_array)))
    # waist_mean = round(findNearest(waist_array, np.mean(waist_array)))
    # waist_mode = stats.mode(waist_array)[0][0]
    #return waist_median