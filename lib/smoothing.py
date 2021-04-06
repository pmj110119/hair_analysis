import numpy as np
try:
    from scipy.fft import fft, ifft
except:
    from scipy import fft, ifft

from lib.pathEvaluation import evaluate_path as evaluate_angle_diff

def linear_offset(path):
    N = len(path)
    delta = (path[-1, :] - path[0, :]) / (N-1)
    return (path - path[[0], :]) - delta * np.arange(N)[:, np.newaxis]

def inverse_linear_offset(path, start, end):
    N = len(path)
    delta = (end - start) / (N-1)
    return path + start[np.newaxis, :] + delta * np.arange(N)[:, np.newaxis]

def extend(path):
    return np.concatenate( (path, np.flip(-path[1:-1, :], axis = 0) ) )

def inverse_extend(path):
    return path[ : path.shape[0]//2 + 1, :]

def smoothing_(path, k = 5):
    '''
        使用傅里叶描述子, 对非闭合曲线进行平滑滤波。
        该方法详见论文：Ding J J, Chao W L, Huang J D, et al. Anti-symmetric Fourier Descriptor for Non-closed Segments[J].
    '''

    n = len(path)
    if (n - 2 < k):
        return path
    
    start, end = path[0, :], path[-1, :]
    path_closed = extend(linear_offset(path))

    Z = fft(path_closed[:, 0] + 1.j * path_closed[:, 1])

    z = ifft(Z[:k], n = 2 * n - 2)
    smoothed_path = inverse_extend( np.stack( (z.real, z.imag), axis = 1 ) )
    smoothed_path = inverse_linear_offset(smoothed_path, start, end).round().astype(np.int32)
    
    # if  (np.linalg.norm((smoothed_path - path), axis = 1).mean() <= 10) \
    #     and (binary[smoothed_path[:, 0], smoothed_path[:, 1] ].all()):
    #     print("using %d Fourier descriptors" % k)
    return smoothed_path

def correcting_(path, r, c):
    '''
        对路径中超过图片边界的点进行校正。
    '''
    path[ path[:, 0] >= r , 0] = r-1
    path[ path[:, 0] < 0 , 0] = 0
    path[ path[:, 1] >= c , 1] = c-1
    path[ path[:, 1] < 0 , 1] = 0
    return path

def evaluate(binary, path):
    N = len(path)
    if (N == 0):
        return np.inf
    return evaluate_angle_diff(path) + (1. - binary[path[:, 0], path[:, 1]].sum() / N)

def skeleton_smoothing(binary, path, width = 50, dist_thres = 5):
    '''
        每30个点取一个锚点，对两个锚点之间的骨架进行平滑。
        如果平滑后发生断裂，取裂开的其中一段使得其和毛发重复最多。
    '''
    path = np.array(path)
    binary = binary.astype(np.uint8)
    r, c = binary.shape

    N = len(path) 
    result = np.zeros(path.shape, dtype = np.float32)
    for st in [-width // 3 * 2, -width // 3, 0]:
        result += np.concatenate( [ smoothing_(path[max(i, 0) : min(i+width, N), :]) for i in range(0, N, width) ] )
    result = (result / 3).round().astype(np.int32)
    result = correcting_(result, r, c)

    dist = np.linalg.norm(result[1:,:] - result[:-1, :], axis = 1) 
    best_seg, best_score, j = result, evaluate(binary, result), 0
    for i in range(N):
        if (i == N-1) or (dist[i] > dist_thres):
            cur_seg = result[j : i+1]
            cur_score = evaluate(binary, cur_seg)
            j = i+1
            if (cur_score < best_score):
                best_score = cur_score
                best_seg = cur_seg
 
    return best_seg
