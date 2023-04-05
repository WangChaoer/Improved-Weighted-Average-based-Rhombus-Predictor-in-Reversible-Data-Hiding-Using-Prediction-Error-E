"""
@预测误差计算
@
@
@
"""
import tool
import numpy as np
import math
from matplotlib import pyplot as plt
from collections import Counter


#针对黑色区域（即 · 集）进行像素值预测
def black_prediction_error(pixels):
    #计算图片的宽和高
    width,height = len(pixels[0]),len(pixels)

    # 计算 ·集 每个像素的预测误差（分为两部分进行计算， · 集以及 X 集合）
    predictValue = np.zeros((height,width))  # 创建一个大小与图片相同的二维数组，用来存放预测值 p
    # average = np.zeros((height,width))  # 创建一个大小与图片相同的二维数组，用来存放平均数 p'
    predictionError = np.zeros((height,width),dtype=int) # 创建一个大小与图片相同的二维数组，用来存放预测误差值 Pe
    predictionErrorlist = []  #用来存放所有的预测误差值
    w1 = np.zeros(6)
    m = np.zeros(6)
    dis = np.zeros(6)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (i + j) % 2 == 0:  # ·集 黑色区域
                a = pixels[i-1][j]
                b = pixels[i][j-1]
                c = pixels[i][j+1]
                d = pixels[i+1][j]
                m[0] = math.floor((a+d) / 2)
                m[1] = math.floor((b+c) / 2)
                m[2] = math.floor((a+b) / 2)
                m[3] = math.floor((a+c) / 2)
                m[4] = math.floor((d+b) / 2)
                m[5] = math.floor((d+c) / 2)
                dis[0] = abs(a - d)
                dis[1] = abs(b - c)
                dis[2]= abs(a - b)
                dis[3] = abs(a - c)
                dis[4] = abs(d - b)
                dis[5] = abs(d - c)
                if sum(dis) == 0 :
                    w1[:] = 1/6
                else:
                    for r in range(len(dis)):
                        w1[r] = math.floor(sum(dis) / (1+dis[r]))

                w = tool.normalization(w1)
                predictValue[i][j] = math.floor(m[0]*w[0]+m[1]*w[1]+m[2]*w[2]+m[3]*w[3]+m[4]*w[4]+m[5]*w[5])
                predictionError[i][j] = pixels[i][j] - predictValue[i][j]
                predictionErrorlist.append(predictionError[i][j])
    #将预测误差转换为int型数组
    predictionErrorlist = np.array(predictionErrorlist)

    # # #计算预测误差值中每个元素出现的个数
    # # cishu = Counter(predictionErrorlist)
    # # print(type(cishu))
    # cishu = tool.count(predictionErrorlist)
    # print(cishu)
    # # 绘制预测误差直方图
    # tool.historgrams(predictionErrorlist)
    # # # tool.max_min(predictionErrorlist)
    # return predictionErrorlist
    # MaxPix,MaxPoint,To_MaxPoint_min_Point,Second_MaxPix,Second_MaxPoint,To_SecondMaxPoint_min_Point=tool.max_and_min(predictionErrorlist,1)
    # print(MaxPix, MaxPoint, To_MaxPoint_min_Point, Second_MaxPix, Second_MaxPoint, To_SecondMaxPoint_min_Point)
    return predictionErrorlist,predictValue

def white_prediction_error(pixels):
    #计算图片的宽和高
    width,height = len(pixels[0]),len(pixels)

    # 计算 x集 每个像素的预测误差（分为两部分进行计算， · 集以及 X 集合）
    predictValue = np.zeros((height,width))  # 创建一个大小与图片相同的二维数组，用来存放预测值 p
    # average = np.zeros((height,width))  # 创建一个大小与图片相同的二维数组，用来存放平均数 p'
    predictionError = np.zeros((height,width),dtype=int) # 创建一个大小与图片相同的二维数组，用来存放预测误差值 Pe
    predictionErrorlist = []  #用来存放所有的预测误差值

    w1 = np.zeros(6)
    m = np.zeros(6)
    dis = np.zeros(6)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (i + j) % 2 != 0:  # ·集 黑色区域
                a = pixels[i - 1][j]
                b = pixels[i][j - 1]
                c = pixels[i][j + 1]
                d = pixels[i + 1][j]
                m[0] = math.floor((a + d) / 2)
                m[1] = math.floor((b + c) / 2)
                m[2] = math.floor((a + b) / 2)
                m[3] = math.floor((a + c) / 2)
                m[4] = math.floor((d + b) / 2)
                m[5] = math.floor((d + c) / 2)
                dis[0] = abs(a - d)
                dis[1] = abs(b - c)
                dis[2] = abs(a - b)
                dis[3] = abs(a - c)
                dis[4] = abs(d - b)
                dis[5] = abs(d - c)
                if sum(dis) == 0:
                    w1[:] = 1/6
                else:
                    for r in range(len(dis)):
                        w1[r] = math.floor(sum(dis) / (1 + dis[r]))
                w = tool.normalization(w1)
                predictValue[i][j] = math.floor(
                    m[0] * w[0] + m[1] * w[1] + m[2] * w[2] + m[3] * w[3] + m[4] * w[4] + m[5] * w[5])
                predictionError[i][j] = pixels[i][j] - predictValue[i][j]
                predictionErrorlist.append(predictionError[i][j])
    #将预测误差转换为int型数组
    predictionErrorlist = np.array(predictionErrorlist)

    #计算预测误差值中每个元素出现的个数
    # cishu = Counter(predictionErrorlist)
    # print(type(cishu))
    # cishu = tool.count(predictionErrorlist)
    # print(cishu)
    # # print("区域B各像素出现的次数分别为",cishu)
    # # 绘制预测误差直方图
    # tool.historgrams(predictionErrorlist)
    # tool.max_min(predictionErrorlist)
    return predictionErrorlist,predictValue

# black_prediction_error("../img/yinzhaoxia/lena.tiff")
# white_prediction_error("../img/yinzhaoxia/lena.tiff")