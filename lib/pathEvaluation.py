import numpy as np
from numba import jit

@jit(nopython = True)
def evaluate_path(path, local_angle_thres = np.pi / 2):
    """
        打分估计`path`的形状是否像一根毛发。越小越像。
    """
    path = path.astype(np.float32)
    N = len(path)
    if (N <= 25):
        return np.inf
    dirs = path[10:, :] - path[:-10, :]

    x1, y1, x2, y2 = dirs[:-10, 0], dirs[:-10, 1], dirs[10:, 0], dirs[10:, 1]
    angle_diff = np.arccos((x1 * x2 + y1 * y2) / np.sqrt((x1*x1 + y1*y1) * (x2*x2 + y2*y2)))
    if (angle_diff.max() >= local_angle_thres ):
        return np.inf

    x1, y1, x2, y2 = dirs[:-1, 0], dirs[:-1, 1], dirs[1:, 0], dirs[1:, 1]
    angle_diff = np.arccos((x1 * x2 + y1 * y2) / np.sqrt((x1*x1 + y1*y1) * (x2*x2 + y2*y2)))
    return angle_diff.std()
