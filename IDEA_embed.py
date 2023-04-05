'''
嵌入步骤：
一、根据白色区域（B）计算黑色区域（A）的波动值
二、根据白色区域（B）计算黑色区域（A）的预测误差
三、将波动值序列进行排序，并将预测误差序列按照波动值序列来排序
四、预测误差值生成预测误差直方图，并寻找两队峰值以及零点值
五、移动直方图
六、嵌入数据
'''
import numpy as np
from collections import Counter
import Prediction_Error
import tool
import Fluctuated_Value

from matplotlib import pyplot as plt
import cv2 as cv

def adaptembed_A(pixels,secret_message):
    #计算图片的宽和高
    width,height = len(pixels[0]),len(pixels)

    capacity = len(secret_message)

    fluctuatedValue = Fluctuated_Value.black_calculate_fluctuated_value(pixels)
    predictionError,predictValue = Prediction_Error.black_prediction_error(pixels)
    sortFluctuatedValue,sortPredictionError = tool.sort(fluctuatedValue,predictionError)
    MaxPix, pk1, z1, Second_MaxPix, pk2, z2 = tool.max_and_min(predictionError, 0)
    # print("区域A第一峰值的像素值为：",pk1,"，共出现：",MaxPix,"次，对应0点的像素值为：",z1)
    # print("区域A第二峰值的像素值为：",pk2, "，共出现：",Second_MaxPix, "次，对应0点的像素值为：",z2)
    # print(MaxPix+Second_MaxPix)
    # maxPayload = pk1 + pk2
    # min_err = int(min(predictionError))
    # max_err = int(max(predictionError))
    # print(pk1, z1, pk2, z2)
    # print(MaxPix,Second_MaxPix)

    # #画预测误差直方图
    # tool.generate_hist(embedarr)
    # plt.show()

    # #移动直方图
    # shift = np.array(embedarr,dtype=int)
    # for i in range(len(shift)):
    #     if z1 < shift[i] < pk1 :
    #         shift[i] -=1
    #     elif pk2 < shift[i] < z2 :
    #         shift[i] += 1

    # #画平移一个像素的直方图
    # tool.generate_hist(shift)
    # plt.show()
    # # print(tool.generate_hist(shift).tolist())

    #挨个扫描并将 信息嵌入, 其余 移位
    T = 0
    n = 0

    for i in range(len(sortPredictionError)):
        if n == (len(secret_message)):
            break
        if sortPredictionError[i] == pk1 :   #按照复杂度排序后如果预测误差是左边的峰值点便减去秘密信息
            sortPredictionError[i] -= secret_message[n]
            n += 1
        elif sortPredictionError[i] == pk2:   #按照复杂度排序后如果预测误差是右边的峰值点便加上秘密信息
            sortPredictionError[i] += secret_message[n]
            n += 1
        elif z1<sortPredictionError[i]<pk1 :
            sortPredictionError[i] -= 1
            T += 1
        elif pk2<sortPredictionError[i]<z2 :
            sortPredictionError[i] += 1
            T += 1

    # print("无效移位",T)

    # print("实际嵌入：",n)

    # #画嵌入秘密消息后的直方图
    # histogram = tool.generate_hist(err_I1)
    # plt.show()
    # histogram_list = histogram.tolist()
    # # print(histogram_list)
    # print(T)


    #恢复预测误差值序列
    recoverpe = tool.sort_recover(fluctuatedValue,sortPredictionError)
    #恢复成二维数组中对应区域A的像素值
    recoverpe2x2= tool.recover_black2x2(recoverpe)
    stegoA = np.array(pixels)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (i + j) % 2 == 0:  # ·集 黑色区域
                stegoA[i][j] = predictValue[i][j] + recoverpe2x2[i][j]

    return stegoA,pk1,z1,pk2,z2

def adaptembed_B(pixels,secret_message,stego_Image):
    #计算图片的宽和高
    width,height = len(pixels[0]),len(pixels)

    capacity = len(secret_message)
    T = 0
    fluctuatedValue = Fluctuated_Value.white_calculate_fluctuated_value(pixels)
    predictionError,predictValue = Prediction_Error.white_prediction_error(pixels)
    sortFluctuatedValue,sortPredictionError = tool.sort(fluctuatedValue,predictionError)

    MaxPix, pk1, z1, Second_MaxPix, pk2, z2 = tool.max_and_min(predictionError, 0)

    # print("区域B第一峰值的像素值为：",pk1,"，共出现：",MaxPix,"次，对应0点的像素值为：",z1)
    # print("区域B第二峰值的像素值为：",pk2, "，共出现：",Second_MaxPix, "次，对应0点的像素值为：",z2)
    # print(MaxPix+Second_MaxPix)
    # maxPayload = pk1 + pk2
    # min_err = int(min(predictionError))
    # max_err = int(max(predictionError))
    # print(pk1, z1, pk2, z2)
    # print(MaxPix,Second_MaxPix)
    #
    # #画预测误差直方图
    # tool.generate_hist(embedarr)
    # plt.show()
    #
    # #移动直方图
    # shift = np.array(embedarr,dtype=int)
    # for i in range(len(shift)):
    #     if z1 < shift[i] < pk1 :
    #         shift[i] -=1
    #     elif pk2 < shift[i] < z2 :
    #         shift[i] += 1
    #
    # #画平移一个像素的直方图
    # tool.generate_hist(shift)
    # plt.show()
    # # print(tool.generate_hist(shift).tolist())

    #挨个扫描并将 信息嵌入, 其余 移位
    T = 0
    n = 0
    for i in range(len(sortPredictionError)):
        if n == (len(secret_message)):

            break
        if sortPredictionError[i] == pk1 :   #按照复杂度排序后如果预测误差是左边的峰值点便减去秘密信息
            sortPredictionError[i] -= secret_message[n]
            n += 1
        elif sortPredictionError[i] == pk2:   #按照复杂度排序后如果预测误差是右边的峰值点便加上秘密信息
            sortPredictionError[i] += secret_message[n]
            n += 1
        elif z1<sortPredictionError[i]<pk1 :
            sortPredictionError[i] -= 1
            T += 1
        elif pk2<sortPredictionError[i]<z2 :
            sortPredictionError[i] += 1
            T += 1
    # print("无效移位",T)

    # print(T,n)

    # print(T)
    # print("实际嵌入：",n)

    # #画嵌入秘密消息后的直方图
    # histogram = tool.generate_hist(err_I1)
    # plt.show()
    # histogram_list = histogram.tolist()
    # # print(histogram_list)


    #恢复预测误差值序列
    recoverpe = tool.sort_recover(fluctuatedValue,sortPredictionError)
    #恢复成二维数组中对应区域A的像素值
    recoverpe2x2= tool.recover_white2x2(recoverpe)
    stegoB = np.array(pixels)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (i + j) % 2 != 0:  # x集 白色区域
                stegoB[i][j] = predictValue[i][j] + recoverpe2x2[i][j]

    # cv.imwrite(stego_Image, stegoB, [cv.IMWRITE_PNG_COMPRESSION, 0])
    cv.imwrite(stego_Image, stegoB, [cv.IMWRITE_PNG_COMPRESSION, 0])
    return stegoB,pk1,z1,pk2,z2
