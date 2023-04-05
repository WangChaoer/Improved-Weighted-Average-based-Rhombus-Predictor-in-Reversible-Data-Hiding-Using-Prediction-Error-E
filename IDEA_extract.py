import numpy as np
import cv2 as cv

import Fluctuated_Value
import Prediction_Error
import tool



#提取需要获得原直方图峰值以及次峰值及其对应的像素值、对应的0点像素值
def extract_A(pixels,pk1,z1,pk2,z2,capacity,origimg,locationmap):

    #计算图片的宽和高
    width,height = len(pixels[0]),len(pixels)


    fluctuatedValue = Fluctuated_Value.black_calculate_fluctuated_value(pixels)
    predictionError,predictValue = Prediction_Error.black_prediction_error(pixels)
    sortFluctuatedValue,sortPredictionError = tool.sort(fluctuatedValue,predictionError)

    secret_message = np.zeros(capacity)
    T = 0  #37390
    n = 0

    pe = np.array(sortPredictionError)
    for i in range(len(pe)):
        T += 1
        if n == (capacity):
            T = T - 1
            break
        if pe[i] == pk1 :
            secret_message[n] = 0
            n += 1
        elif pe[i] == (pk1-1) :   #按照复杂度排序后如果预测误差是左边的峰值点便减去秘密信息
            secret_message[n] = 1
            pe[i] += 1
            n += 1
        elif pe[i] == pk2:   #按照复杂度排序后如果预测误差是右边的峰值点便加上秘密信息
            secret_message[n] = 0
            n += 1
        elif pe[i] == (pk2+1):   #按照复杂度排序后如果预测误差是右边的峰值点便加上秘密信息
            secret_message[n] = 1
            pe[i] -= 1
            n += 1
        elif z1<=pe[i]<(pk1-1) :
            pe[i] += 1
        elif (pk2+1)<pe[i]<=z2 :
            pe[i] -= 1

    # print(T)


    #恢复预测误差值序列
    recoverpe = tool.sort_recover(fluctuatedValue,pe)
    #恢复成二维数组中对应区域A的像素值
    recoverpe2x2= tool.recover_black2x2(recoverpe)
    origB = np.array(pixels)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (i + j) % 2 == 0:
                origB[i][j] = predictValue[i][j] + recoverpe2x2[i][j]
    origB = tool.recoverflow(origB,locationmap)
    cv.imwrite(origimg, origB)
    return secret_message



#提取需要获得原直方图峰值以及次峰值及其对应的像素值、对应的0点像素值
def extract_B(image,pk1,z1,pk2,z2,capacity):

    #计算图片的宽和高
    width,height = tool.get_w_h(image)
    pixels = tool.img_to_array(image)

    fluctuatedValue = Fluctuated_Value.white_calculate_fluctuated_value(pixels)
    predictionError,predictValue = Prediction_Error.white_prediction_error(pixels)
    sortFluctuatedValue,sortPredictionError = tool.sort(fluctuatedValue,predictionError)

    secret_message = np.zeros(capacity)
    T = 0  #37390
    n = 0

    pe = np.array(sortPredictionError)
    for i in range(len(pe)):
        T += 1
        if n == (capacity):
            T = T - 1
            break
        if pe[i] == pk1 :
            secret_message[n] = 0
            n += 1
        elif pe[i] == (pk1-1) :   #按照复杂度排序后如果预测误差是左边的峰值点便减去秘密信息
            secret_message[n] = 1
            pe[i] += 1
            n += 1
        elif pe[i] == pk2:   #按照复杂度排序后如果预测误差是右边的峰值点便加上秘密信息
            secret_message[n] = 0
            n += 1
        elif pe[i] == (pk2+1):   #按照复杂度排序后如果预测误差是右边的峰值点便加上秘密信息
            secret_message[n] = 1
            pe[i] -= 1
            n += 1
        elif z1<=pe[i]<(pk1-1) :
            pe[i] += 1
        elif (pk2+1)<pe[i]<=z2 :
            pe[i] -= 1

    # print(T)


    #恢复预测误差值序列
    recoverpe = tool.sort_recover(fluctuatedValue,pe)
    #恢复成二维数组中对应区域A的像素值
    recoverpe2x2= tool.recover_white2x2(recoverpe)
    origB = np.array(pixels)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if (i + j) % 2 != 0:  # ·集 黑色区域
                origB[i][j] = predictValue[i][j] + recoverpe2x2[i][j]

    return secret_message,origB

