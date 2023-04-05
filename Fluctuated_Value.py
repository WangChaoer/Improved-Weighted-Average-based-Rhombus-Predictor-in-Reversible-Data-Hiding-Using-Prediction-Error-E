'''
@方差以及波动值计算
@实验部分
@主要内容：计算像素值的方差，并在方差的基础上计算除第一行和最后一行以及第一列和最后一列像素的波动值
'''
import numpy as np
import cv2
import tool

#计算方差


#计算黑色集（A集 ·集）每个像素的波动值（除去第一行第一列以及最后一行最后一列）
def black_calculate_fluctuated_value(pixels):

    #计算图片的宽和高
    width,height = len(pixels[0]),len(pixels)

    #波动值数组的定义（不计算第一行和最后一行以及第一列和最后一列）
    flucatedValue = np.zeros((height,width))
    #波动值的计算（不计算第一行和最后一行以及第一列和最后一列）
    flucatedValueList = []
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (i + j) % 2 == 0:
                a = pixels[i-1][j]
                b = pixels[i][j-1]
                c = pixels[i][j+1]
                d = pixels[i+1][j]
                flucatedValue[i][j] = abs(a-d)+abs(b-c)+abs(a-b)+abs(a-c)+abs(d-b)+abs(d-c)
                flucatedValueList.append(flucatedValue[i][j])
    flucatedValueList = np.array(flucatedValueList)

    # print(flucatedValue)
    return flucatedValueList

#计算白色集（B集 x集）每个像素的波动值（除去第一行第一列以及最后一行最后一列）
def white_calculate_fluctuated_value(pixels):

    #计算图片的宽和高
    width,height = len(pixels[0]),len(pixels)
    #获取图像每个像素点的方差

    #波动值数组的定义（不计算第一行和最后一行以及第一列和最后一列）
    flucatedValue = np.zeros((height,width))
    #波动值的计算（不计算第一行和最后一行以及第一列和最后一列）
    flucatedValueList = []
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (i + j) % 2 != 0:
                a = pixels[i-1][j]
                b = pixels[i][j-1]
                c = pixels[i][j+1]
                d = pixels[i+1][j]
                flucatedValue[i][j] = abs(a-d)+abs(b-c)+abs(a-b)+abs(a-c)+abs(d-b)+abs(d-c)
                flucatedValueList.append(flucatedValue[i][j])
    flucatedValueList = np.array(flucatedValueList)
    # print(flucatedValue)
    return flucatedValueList
