from ultralytics import YOLO

import cv2
import numpy as np
import os

num_0 = 0
num_1 = 0
num_2 = 0
num_3 = 0
num_4 = 0
num_5 = 0
num_6 = 0
num_7 = 0
num_8 = 0

for i in os.listdir('Input'):
    # 加载模型
    model = YOLO('best.pt')

    # 读取图片
    src = cv2.imread(f'Input/{i}')

    # 灰度化
    gray_image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Canny边缘检测
    edges = cv2.Canny(gray_image, 100, 200)

    # 轮廓检测
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    cv2.drawContours(src, contours, -1, (0, 0, 255), 2)

    # 进行图像预测
    res = model(src)

    # 对预测结果进行统计
    for r in res:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs

    for r in res:
        boxes = r.boxes  # Boxes object for bbox outputs

        class_ids = boxes.cls.cpu().numpy().astype(int)  # 转为int类型数组
        num = np.sum(class_ids == 0)  # 对类别为0的检测框数目作一个统计
        num_0 += num
        num = np.sum(class_ids == 1)  # 对类别为1的检测框数目作一个统计
        num_1 += num
        num = np.sum(class_ids == 2)  # 对类别为2的检测框数目作一个统计
        num_2 += num
        num = np.sum(class_ids == 3)  # 对类别为3的检测框数目作一个统计

        # 判断是否破损
        if num != 0:
            print("第", i, "张图片中的天花板有破损")

        num_3 += num
        num = np.sum(class_ids == 4)  # 对类别为4的检测框数目作一个统计
        num_4 += num
        num = np.sum(class_ids == 5)  # 对类别为5的检测框数目作一个统计
        num_5 += num
        num = np.sum(class_ids == 6)  # 对类别为6的检测框数目作一个统计
        num_6 += num
        num = np.sum(class_ids == 7)  # 对类别为7的检测框数目作一个统计
        num_7 += num

    # 绘制结果
    annotated_img = res[0].plot()
    cv2.imwrite(filename=f'Output/{i}', img=annotated_img)

print("数量统计：")
print("物体A:", num_0)
print("物体B:", num_1)
print("物体C:", num_2)
print("物体D:", num_3)
print("灯:", num_4)
print("烟雾报警器:", num_5)
print("路由器:", num_6)
print("应急灯:", num_7)
