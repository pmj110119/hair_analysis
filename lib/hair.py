import cv2
import numpy as np

import matplotlib.pyplot as plt
from math import *
from PIL import Image,ImageDraw
import scipy as sp
from scipy.interpolate import splprep  # 增加该行
def getOrientation(pts):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    

    if(len(eigenvectors)<2):
        return 1000
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians #PCA第一维度的角度    
    return angle



def get_PCA_angle(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    areas=[]
    areas_angle=[]
    for i, c in enumerate(contours):
        # 面积筛选,暂时不用    后面用百分比来滤
        area = cv2.contourArea(c)
        if area < 8:
           # print('面积太小：',area)
            continue
        rect = cv2.minAreaRect(c)
        [w,h] = rect[1]
        if(w>h):
            temp = w
            w = h
            h = temp
        if(h/w<2.5):
            continue
        #cv2.drawContours(src, contours, i, (0, 0, 255), 2)
        #plt.figure(1);plt.imshow(src,cmap='gray')
        angle=getOrientation(c)    # Find the orientation of each shape
        if(angle==1000):
            continue
        areas.append(area)
        areas_angle.append(angle)
    
    #plt.figure(2);plt.imshow(src,cmap='gray')
    if(len(areas)==0):
        return None
    ind=np.argmax(areas)
 
    return areas_angle[ind]*57.3


# 裁剪旋转矩形中的像素
def getWarpTile(img,center,rect_width,rect_height,rect_angle, inverse=False):
    [xp_Click,yp_Click] = center
    ysize,xsize = img.shape[:2]
    # 计算四个角点坐标
    p1 = [round(max(0,xp_Click-rect_height//2)),round(max(0,yp_Click-rect_width//2))]
    p2 = [round(max(0,xp_Click-rect_height//2)),round(min(ysize,yp_Click+rect_width//2))]
    p3 = [round(min(xsize,xp_Click+rect_height//2)),round(min(ysize,yp_Click+rect_width//2))]
    p4 = [round(min(xsize,xp_Click+rect_height//2)),round(max(0,yp_Click-rect_width//2))]
    xg = np.array([p1[0],p2[0],p3[0],p4[0]])
    yg = np.array([p1[1],p2[1],p3[1],p4[1]])
    xg_t = xp_Click + (xg-xp_Click)*cos(rect_angle) + (yg-yp_Click)*sin(rect_angle) # 旋转
    yg_t = yp_Click + (yg-yp_Click)*cos(rect_angle) - (xg-xp_Click)*sin(rect_angle)
    cnt = np.array([
        [[int(xg_t[0]), int(yg_t[0])]],
        [[int(xg_t[1]), int(yg_t[1])]],
        [[int(xg_t[2]), int(yg_t[2])]],
        [[int(xg_t[3]), int(yg_t[3])]]
    ])
    
    # 得到旋转后的box
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width = int(rect[1][0])
    height = int(rect[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    tile = cv2.warpPerspective(img, M, (width, height))
    
    # 90度校正——不加的话会有图片横竖不一的bug
    if(inverse==False):
        if(tile.shape[0]>tile.shape[1]):
            tile = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
            [[cX,cY],[w,h],angle] = rect
            rect = ((cX,cY),(h,w),angle+90)
    else:
        if(tile.shape[0]<tile.shape[1]):
            tile = cv2.rotate(tile, cv2.ROTATE_90_CLOCKWISE)
            [[cX,cY],[w,h],angle] = rect
            rect = ((cX,cY),(h,w),angle+90)

    return {'tile':tile,'rect':rect, 'box':box}


def get_width(tile_binary,threshold=150):

    contours, _ = cv2.findContours(tile_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    area_max = 0
    cnt = []
    for i, c in enumerate(contours):  # 只考虑最大轮廓
        area = cv2.contourArea(c)
        if(area>area_max):
            area_max = area
            cnt = c
    if(len(cnt)==0):
        return [0,0] 
    else:
        x, y, w, h = cv2.boundingRect(cnt)         #用一个矩形将轮廓包围
        return [h,(y+h/2)-tile_binary.shape[0]/2.0]


def length_correct(init_rect, curve_binary):
    # direction为1时向左拓展，direction为-1时向右拓展
    def expand_box(rect, direction, w_add):

        [[x, y], [w, h], angle] = rect

        # print(angle)
        full_x = x + direction * w_add / 2 * cos(angle / 57.3)
        full_y = y + direction * w_add / 2 * sin(angle / 57.3)
        full_w = w + abs(direction) * w_add
        full_rect = ((full_x, full_y), (full_w, h), angle)

        head_x = x + (direction * w / 2 + direction * w_add / 2) * cos(angle / 57.3)

        # print(w,(w/2.0 + direction * w_add/2.0),cos(angle/57.3))
        head_y = y + (direction * w / 2 + direction * w_add / 2) * sin(angle / 57.3)
        head_w = abs(direction) * w_add

        head_rect = ((head_x, head_y), (head_w/2, h), angle)
        # print(rect,'...',head_rect)
        return {'full': full_rect, 'head': head_rect}

    # 向左拓展
    for width_add in range(30):
        # rect拓展
        expand_result = expand_box(init_rect, -1, 5)
        rect_full = expand_result['full']
        rect_head = expand_result['head']
        init_rect = rect_full
        # 提取头部rect图像
        [[x, y], [w, h], angle] = rect_head
        tile_result = getWarpTile(curve_binary, [x, y], h, w, (180 - angle) / 57.3, inverse=False)
        tile = tile_result['tile']
        # 计算白色比例
        ratio = tile.sum() / 255 / (tile.shape[0] * tile.shape[1])
        # 比例小于阈值，跳出
        if (ratio > 0.7):
            init_rect = rect_full
        else:
            break
    # 向右拓展
    rect_full = init_rect
    for width_add in range(30):
        # rect拓展
        expand_result = expand_box(rect_full, 1, 5)
        rect_full = expand_result['full']
        rect_head = expand_result['head']

        # 提取头部rect图像
        [[x, y], [w, h], angle] = rect_head
        tile_result = getWarpTile(curve_binary, [x, y], h, w, (180 - angle) / 57.3, inverse=False)
        tile = tile_result['tile']
        # 计算白色比例
        ratio = tile.sum() / 255 / (tile.shape[0] * tile.shape[1])
        # 比例小于阈值，跳出

        if (ratio > 0.7):
            init_rect = rect_full
        else:
            break


    return init_rect


def auto_search(img_binary,center,rect_width,rect_height,is_length_correct=False):
   # cv2.imwrite('img.png', img_binary)
    ## 1.角度校正
    FOUND_ANGLE = False
    rotate_angle_corrected = None
    visual_height = None
    for rotate_angle in range(0,180):
        result = getWarpTile(img_binary,center, rect_width, rect_height, rect_angle=rotate_angle/57.3)
        
        tile = result['tile']
        rect = result['rect']
        visual_height = rect[1][1]
        box = result['box']
        # 计算主成分角度
        angle = get_PCA_angle(tile)



        if(angle!=None):
            if(angle<1 and angle>-1 ):
                FOUND_ANGLE = True
                rotate_angle_corrected = rotate_angle
                rect_rotate_corrected = rect
                break
    if(FOUND_ANGLE==False):
        return {'is_find':False, 'box':None}
    

    ## 2.中心偏移校正    (不需要卡的很死，只要保证毛发整体进入视野就行)
    Y_diff = 100
    new_center = center
    niter = 0
    while(abs(Y_diff)>0.5):
        # 计算当前box最大轮廓的质心坐标               # 这里30是为了缩短宽度，减少影响，但不能比width更短，不然rect会反90度（因为warp时强行设置width>height了，后面修正）
        result = getWarpTile(img_binary,new_center, rect_width, int(rect_width*1.5), rect_angle=rotate_angle_corrected/57.3)
        tile = result['tile']
        rect = result['rect']


        rect = (rect[0],(rect[1][0],visual_height),rect_rotate_corrected[2])

        rect_origin = rect

        box = cv2.boxPoints(rect)
        box = np.int0(box)



        contours, _ = cv2.findContours(tile, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        area_max = 0
        cY = None
        for i, c in enumerate(contours):  # 只考虑最大轮廓
            area = cv2.contourArea(c)
            if(area>area_max):
                area_max = area
                rect = cv2.minAreaRect(c)
                [[cX,cY],[w,h],angle] = rect
                up = cY + h/2
                down = cY - h/2
        if(cY == None):
            return {'is_find':False, 'box':None}



        # 反馈控制————单P
        Y_diff = cY - rect_width/2
        new_center[0] += Y_diff*cos(angle)*0.5
        new_center[1] += Y_diff*sin(angle)*0.5


        # 异常
        niter += 1
        if(niter>40 ):
            #if(abs(Y_diff)>1.0):
            print('    errror:超过40次')
            return {'is_find':False, 'box':None}

    ## 3.宽度校正    原本20x70    改为20x5，用这一小截来确定宽度
    result = getWarpTile(img_binary,new_center, rect_width, 5, rect_angle=rotate_angle_corrected/57.3,inverse=True)
    tile = result['tile']
    # 测宽
    [height,y_offset] = get_width(tile)
    if(height <= 0):
        return {'is_find': False, 'rect': None, 'box': None}
    [xx, yy] = rect_origin[0]
    rect = ((xx, yy + y_offset), (rect_rotate_corrected[1][0], height), rect_origin[2])


    ## 4.长度校正
    if(is_length_correct==True):
        rect = length_correct(rect,img_binary)

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return {'is_find': True, 'rect': rect, 'box': box}



def impaint(results,src):
    img = src.copy()
    mask = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    # mask_expand = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for result in results:
        joint = result['joints']
        width = result['width']
        for i in range(len(joint) - 1):
            p1 = (int(joint[i][0]), int(joint[i][1]))
            p2 = (int(joint[i + 1][0]), int(joint[i + 1][1]))
            cv2.line(img, p1, p2, (255, 255, 255), width+5, cv2.LINE_AA)
            cv2.line(mask, p1, p2, (255, 255, 255), width+5, cv2.LINE_AA)
    h,w = img.shape[:2]
    if w>1000:
        img = cv2.resize(img,(w//2,h//2))
        mask = cv2.resize(mask, (w // 2, h // 2))
    output = cv2.inpaint(img, mask, 5, flags=cv2.INPAINT_TELEA)
    output = cv2.resize(output,(w,h))
    return output


def getMidPoint(joint):
    length = np.zeros(len(joint) - 1)
    for i in range(len(joint) - 1):
        distance = sqrt((int(joint[i][0] - joint[i + 1][0]) ** 2) + (int(joint[i][1] - joint[i + 1][1]) ** 2))
        length[i] = distance

    length_sum = np.cumsum(length)
    mid_sum = length.sum()/2.0
    idx = np.searchsorted(length_sum,mid_sum)
    if idx==0:
        ratio = mid_sum/length[idx]
    else:
        ratio = (mid_sum-length_sum[idx-1]) / length[idx]
    x = joint[idx][0] + (joint[idx+1][0]-joint[idx][0]) * ratio
    y = joint[idx][1] + (joint[idx+1][1]-joint[idx][1]) * ratio
    return [x,y]

def fitCurve(joint):
    k=3
    if len(joint)<=3:
        return joint

    x = np.zeros(len(joint))
    y = np.zeros(len(joint))
    for idx,joint in enumerate(joint):
        x[idx] = joint[0]
        y[idx] = joint[1]
    tcktuples, uarray = sp.interpolate.splprep([x, y],k=k,s=0)
    unew = np.arange(0, 1.02, 0.02)
    splinevalues = sp.interpolate.splev(unew, tcktuples)
    new_joint = []
    for x,y in zip(splinevalues[0],splinevalues[1]):
        new_joint.append([x,y])
    return new_joint

def curve_plot(img,results,distinguishValue=0,color1=(0, 200, 150),color2=(0, 100, 200),alpha=1,roi=None,handle_diff=None,handle_width=None):
    if roi is not None:
        x0, x1, y0, y1 = roi
        img = img[y0:y1, x0:x1]
    else:
        x0,y0=[0,0]
    if handle_diff is not None:
        x0, y0 = handle_diff


    if len(img.shape)==2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    src = img.copy()
    img_color = Image.fromarray(img)
    draw_img = ImageDraw.Draw(img_color)  # 实例化一个ImageDraw

   # curve = np.zeros((img.shape[0],img.shape[1],img.shape[2]),dtype=np.uint8)
   # curve = Image.fromarray(curve)
   # draw_curve = ImageDraw.Draw(curve)  # 实例化一个对象
    for result in results:
        joint = result['joints']
        width = result['width']
        if handle_width is not None:
            width = handle_width
        mid_point = result['mid']
        joint_fit = fitCurve(joint)

        if width>=distinguishValue:
            color=color2
        else:
            color=color1
        # 画线
        for i in range(len(joint_fit) - 1):
            p1 = (int(joint_fit[i][0])-x0, int(joint_fit[i][1])-y0)
            p2 = (int(joint_fit[i + 1][0])-x0, int(joint_fit[i + 1][1])-y0)
            draw_img.line(p1 + p2 , fill=color, width=width)

        radius = width / 2
        if radius<3:
            radius=3
        if handle_diff is not None:
            # 画关节点
            for i in range(len(joint)):
                p1 = (int(joint[i][0]), int(joint[i][1]))
                draw_img.ellipse((p1[0] - radius-x0, p1[1] - radius-y0, p1[0] + radius-x0, p1[1] + radius-y0), fill=(0, 255, 0), outline=(0, 0, 0))
            # 画中点
            draw_img.ellipse((mid_point[0] - radius-x0, mid_point[1] - radius-y0, mid_point[0] + radius-x0, mid_point[1] + radius-y0), fill=(255, 255, 0), outline=(0, 0, 0))



    img_color = np.array(img_color).astype(np.uint8)    # PIL转回numpy
    # curve = np.array(curve).astype(np.uint8)    # PIL转回numpy


    if len(img_color.shape) == 2:
        img_color = cv2.cvtColor(img_color, cv2.COLOR_GRAY2BGR)
    overlapping = cv2.addWeighted(img_color, alpha, src, 1-alpha, 0)
    return {'img':overlapping}




if __name__ == "__main__":

    img = cv2.imread('imgs/mini_2.jpg')
    curve = img.copy()
    result=[]
    
    # 鼠标双击的回调函数
    def action(event,x_click,y_click,flags,param):
        global result
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # process
            pkg = auto_search(curve,[x_click,y_click],20,60,binary_threshold=140)
            # 保存结果
            is_find = pkg['is_find']
            if(is_find):  
                box = pkg['box']
                rect = pkg['rect']
                #print('rect:',rect)
                result.append({'rect':rect, 'box':box, 'width':rect[1][1]})

    cv2.namedWindow('image')
    cv2.setMouseCallback('image',action)
    while(True):
        temp = curve_plot(img,result)
        curve = temp['img']
        cv2.imshow('image',curve)
        command = cv2.waitKey(1)
        if(command==27):
            cv2.destroyAllWindows()
            break
        elif(command==100): # 'd'
            result.pop(-1)
            print(len(result))
        elif(command==115): # 's'
            print(len(result))
            width = []
            for result_ in result:
                width.append(result_['width'])
            plt.hist(np.array(width))
            plt.figure(1)
            plt.show()
        elif(command!=-1):
            print(command)

        

        