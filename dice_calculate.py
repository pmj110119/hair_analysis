import numpy as np
import cv2
from lib.hair import curve_plot,getMidPoint,fitCurve
import json
import os
from math import *
from lib.utils import *
from PIL import Image,ImageDraw
import glob

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def loadJson(jsonPath):
    result = []
    result_origin =[]
    if os.path.exists(jsonPath):
        with open(jsonPath, 'r') as f:
            datas = json.load(f)

        for data in datas:
            if len(data['joints'])<2:
                continue
            if data['width']<=1 or isnan(data['mid'][0]):
                continue


            d={}
            d_origin={}
            joints = data['joints']

            joints_origin = []
            for joint in joints:
                joints_origin.append(joint.copy())
                joint[0] = int(joint[0])
                joint[1] = int(joint[1])
            d['joints'] = joints
            d_origin['joints'] = joints_origin


            d_origin['width'] = data['width']
            d['width'] = int(data['width'])

            if 'mid' in data.keys():
                d_origin['mid'] = data['mid'].copy()
                mid = data['mid']
                mid[0] = int(mid[0])
                mid[1] = int(mid[1])
                d['mid'] = mid
            else:
                d['mid'] = getMidPoint(data['joints'])

            result.append(d)
            result_origin.append(d_origin)
    return result,result_origin





def curve_plot(img,results,color=(255)):

    x0,y0=[0,0]

    img_color = Image.fromarray(img)
    draw_img = ImageDraw.Draw(img_color)  # 实例化一个ImageDraw

    for result in results:
        joint = result['joints']
        if len(joint)<2:
            continue
        width = result['width']
        mid_point = result['mid']
        joint_fit = fitCurve(joint)    # 暂时关闭曲线拟合功能

        # 画线
        for i in range(len(joint_fit) - 1):
            #print(joint_fit[i])
            try:        # IndexError: invalid index to scalar variable.
                p1 = (int(joint_fit[i][0])-x0, int(joint_fit[i][1])-y0)
                p2 = (int(joint_fit[i + 1][0])-x0, int(joint_fit[i + 1][1])-y0)
                draw_img.line(p1 + p2 , fill=color, width=int(width))
            except:
                continue



    img_color = np.array(img_color).astype(np.uint8)    # PIL转回numpy
    return img_color



jsons = glob.glob('data/imgs/*.json')
for pred_json_path in jsons:
    pred_results ,_ = loadJson(pred_json_path)
    if len(pred_results)==0:
        continue
    label_results ,_ = loadJson('../saved/'+os.path.basename(pred_json_path))

    match_num = 0
    for p_index,pred in enumerate(pred_results):
        pred_curve = np.zeros((2048,2560),np.uint8)
        pred_curve = curve_plot(pred_curve,[pred],color=1)
        pred_mid = pred['mid']
        for label in label_results:
            label_mid = label['mid']
            if abs(label_mid[0]-pred_mid[0])>50 or abs(label_mid[1]-pred_mid[1])>50:
                continue
            label_curve = np.zeros_like(pred_curve, np.uint8)
            label_curve = curve_plot(label_curve, [label],color=1)
            dice = dice_coef(label_curve,pred_curve)
            # if dice>0.2:# and dice < 0.4:
            #     label_curve2 = np.zeros((2048,2560,3), np.uint8)
            #     label_curve2 = curve_plot(label_curve2, [label], color=(0,0,255))
            #     label_curve2 = curve_plot(label_curve2, [pred], color=(0, 255, 0))
            #     print(dice)
            #     cv2.imshow('aaa',cv2.resize(label_curve2,(640,512)))
            #     cv2.waitKey(0)
            if dice>0.5:
                match_num = match_num +1
                break
    print(os.path.basename(pred_json_path),'    Precision:',match_num/len(pred_results), '    Recall:',match_num/len(label_results))