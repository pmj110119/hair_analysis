import torch
import numpy as np
import os
import cv2
import gc
import time
import glob
# from networks import se_resnext50_scse,mobilenet
from lib.segmentation.unet import unet_resnet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# device = torch.device('cpu')


def make_grid(shape, window=256, min_overlap=32):
    """
        需注意，如果shape不是window的倍数，实际产生的patch不等于window
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2
    """
    x, y, c = shape  # img.shape
    # 横轴切分
    nx = x // (window - min_overlap)  # 切分数目
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)  # 横轴切分-box左
    x1[-1] = x - window  # 边界情况
    x2 = (x1 + window).clip(0, x)  # box右
    # 纵轴切分
    ny = y // (window - min_overlap)
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx, ny, 4), dtype=np.int64)

    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2: img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


pth_epoch = 710  # 加载之前训练的模型(指定轮数)


def predict_binary(model, img, device):
    slices = make_grid(img.shape, window=512, min_overlap=0)
    h, w, c = img.shape
    curve = np.zeros([h, w], dtype=np.uint8)
    model.eval()
    with torch.no_grad():  # 不进行反向传播
        for index,(x1, x2, y1, y2) in enumerate(slices):
            #print(index/len(slices))
            patch = img[x1:x2, y1:y2, :]
            patch = img2tensor((patch - patch.mean()) / patch.std())
            patch = patch.unsqueeze(0).to(device)
            output = model(patch)  # heatmap
            output = output[0][0].detach().cpu().numpy()  # .astype(np.uint8)*255
            flips = [[-1], [-2]]
            for f in flips:
                x_flip = torch.flip(patch, f)
                output_flip = model(x_flip)
                output_flip = torch.flip(output_flip, f)
                output += output_flip[0][0].detach().cpu().numpy()  # .astype(np.uint8)*255
            output /= (1 + len(flips))
            output = (output > 0.5).astype(np.uint8)
            curve[x1:x2, y1:y2] += output
        curve = (curve > 0.5).astype(np.uint8)
    return curve








def test():
    # model = se_resnext50_scse().to(device)
    model = unet_resnet('resnet34', 3, 3, False).to(device)
    print('创建model成功')

    # pre=torch.load(os.path.join('./checkpoint',str(pth_epoch)+'.pth'), map_location=device)
    # model.load_state_dict(pre)

    model.eval()

    img_dir = 'test'
    fnames = glob.glob(img_dir + '/*.jpg')
    for fname in fnames:
        print(fname)
        src = cv2.imread(fname)
        img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB) / 255.0
        output = predict_binary(model, img) * 255

        cv2.imwrite('result/' + os.path.basename(fname) + '.png', output)


if __name__ == "__main__":
    test()