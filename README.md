## 1. 安装

终端执行命令：

```bash
pip install -r requirements.txt -i https://pypi.douban.com/simple --default-time=100
```

## 2. 运行

```bash
python main.py
```

## 3. 如何调试自己对应的代码

打开py文件 `./lib/process.py`，其中有`BasicProcess`和`MyProcess`两个类

`BasicProcess`中的三个成员函数分别对应栋栋、代超、啸宇的实验（具体说明详见函数注释）:

```python
class BasicProcess():
    def magnet(self,point,img_binary,size = 16): # 在每标注一个点时调用
		pass
    def waist(self,joints,img_binary): 	# 在完成一根毛发标注时调用
		pass
    def border(self,joints,img_binary):	# 在完成一根毛发标注时调用
		pass
```

要调试自己的代码时，在另一个类`MyProcess`中重载`BasicProcess`对应的函数即可。

以栋栋代码为例：

```python
class MyProcess(BasicProcess):
    # 吸铁石
    def magnet(self,point, img_binary, size=16):
        # 确定直线矩阵
        line_cloumn = img_binary[point[0] - size:point[0] + size + 1, point[1]]
        line_row = img_binary[point[0], point[1] - size:point[1] + size + 1]
        # 确定垂直线第一个和最后一个数值最低的点的坐标
        indx = cv2.minMaxLoc(line_cloumn, None)
        indx2 = np.where(line_cloumn == indx[0])
        l0 = indx2[0]  # 所有最小值点的坐标
        l = np.size(indx2, 1)  # 最小值的个数
        firstmin_loc = indx[2]
        firstmin_loc = np.array([firstmin_loc[1], 0])
        lastmin_loc = np.array([l0[l - 1], 0])
        # 确定中间点的坐标
        mid = (firstmin_loc + lastmin_loc) / 2
        mid = np.around(mid)
        if mid[0] == size:
            mid = firstmin_loc
        row = mid[0] - size
        point2_column = np.array([point[0] + row, point[1]])

        # 确定水平线第一个和最后一个数值最低的点的坐标
        indx3 = cv2.minMaxLoc(line_row, None)
        indx4 = np.where(line_row == indx3[0])
        l01 = indx4[0]  # 所有最小值点的坐标
        l11 = np.size(indx4, 1)  # 最小值的个数
        firstmin_loc1 = indx3[2]
        firstmin_loc1 = np.array([0, firstmin_loc1[1]])
        lastmin_loc1 = np.array([0, l01[l11 - 1]])
        # 确定中间点的坐标
        mid1 = (firstmin_loc1 + lastmin_loc1) / 2
        mid1 = np.around(mid1)
        if mid1[1] == size:
            mid1 = firstmin_loc1
        row1 = mid1[1] - size
        point2_column1 = np.array([point[0], point[1] + row1])
        if abs(row) <= abs(row1):
            point2 = point2_column
        else:
            point2 = point2_column1

        return point2
```



# 打包exe

```bash
pyinstaller -F main.py --hidden-import=torchvision --hidden-import=scipy  --paths E:\anaconda3\envs\py36\Lib\site-packages\scipy\.libs

```

