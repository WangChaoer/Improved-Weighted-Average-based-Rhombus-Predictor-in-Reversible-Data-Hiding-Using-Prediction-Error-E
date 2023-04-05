import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
import decimal
from PIL import Image
import math
def generate_random_number(payload):
    secret_info = np.random.randint(0,2,payload) #取[low,high)之间随机整数
    return secret_info

def get_w_h(image):
    img = cv2.imread(image,cv2.IMREAD_GRAYSCALE) #读取灰度图
    width = img.shape[1]
    height = img.shape[0]
    return width,height

def img_to_array(image):    #将图片转换为二维数组
    img = cv2.imread(image,cv2.IMREAD_GRAYSCALE) #读取灰度图
    # cv2.imshow('1', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(img.shape)
    width,height = get_w_h(image)
    pixels = np.zeros((height,width))   # 创建一个大小与图片相同的二维数组
    # pixels = [[0] * width for i in range(height)]  # 创建一个大小与图片相同的二维数组
    img_array = np.array(img)
    for x in range(height):      #   行   高
        for y in range(width):  #   列   宽
            pixels[x][y] = img_array[x,y]
    return  pixels

def normalization(array):
    s = sum(array)
    last = np.zeros(len(array))
    for i in range (len(array)):
        last[i] = array[i] / s
    return last

def softmax(array):
    e = math.e
    last = np.zeros(len(array))
    x = np.zeros(len(array))
    for i in range (len(array)):
        x[i] = e**array[i]
    s = sum(x)
    for j in range(len(array)):
        last[j] = x[j]/s
    return last


#绘制直方图
def historgrams (predictionError):
    # 灰度图像矩阵的高，宽
    maxa = max(predictionError)
    mina = min(predictionError)
    zushu = maxa - mina

    plt.hist(predictionError, bins=zushu)
    plt.xticks(range(mina, maxa, 5))
    plt.xlabel("idea")
    # plt.grid() #网格
    plt.show()

def max_min(predictionError):
    maxa = max(predictionError)
    mina = min(predictionError)
    zushu = maxa - mina
    # 存储灰度直方图
    grayHist = np.zeros(int(zushu), np.uint64)
    for r in range(len(predictionError)):
        grayHist[predictionError[r]] += 1
    # 寻找灰度直方图的最大峰值对应的灰度值
    maxLoc = np.where(grayHist == np.max(grayHist))
    # print(maxLoc)
    firstPeak = maxLoc[0][0] #灰度值
    # 寻找灰度直方图的第二个峰值对应的灰度值
    measureDists = np.zeros([zushu], np.float32)
    for k in range(zushu):
        measureDists[k] = pow(k - firstPeak, 2) * grayHist[k] #综合考虑 两峰距离与峰值
    maxLoc2 = np.where(measureDists == np.max(measureDists))
    secondPeak = maxLoc2[0][0]
    print('双峰为：',firstPeak,secondPeak)

#计算预测误差值中每个元素出现的个数
def count(predictionError):
    Demo_list = predictionError
    Demo_dict = {}
    for key in Demo_list:
        Demo_dict[key] = Demo_dict.get(key, 0) + 1

    sorted_d = sorted(Demo_dict.items(), key=lambda x: x[1])
    # print(type(sorted_d))
    # print(sorted_d)  # Output: [('c', 1), ('a', 2), ('b', 3)]
    return sorted_d

# def count(predictionError):
    # grayHist = np.zeros([256],np.uint64)
    # for i in range(len(predictionError)):
    #     grayHist[int(predictionError[i])] += 1
    # return grayHist

# 修改舍入方式为四舍五入
def halfadjust(predictValue):
    decimal.getcontext().rounding = "ROUND_HALF_UP"
    x = str(predictValue)
    y1 = int(decimal.Decimal(x).quantize(decimal.Decimal("0")))
    return y1

# 计算峰值信噪比的函数
def PSNR(template, img):
    mse = np.mean((template / 255. - img / 255.)**2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

#按照预测误差值排序
def sort(fluctuated_value,prediction_error):
    order = np.argsort(fluctuated_value)
    sort_fluctuated_value = np.sort(fluctuated_value)
    sort_prediction_error = np.zeros_like(fluctuated_value)
    for idx, num in enumerate(prediction_error):
        sort_prediction_error[idx] = prediction_error[order[idx]]
    # print(type(sort_fluctuated_value))
    # print(type(sort_prediction_error))
    return sort_fluctuated_value,sort_prediction_error



#按照排序号的预测误差值 来恢复预测误差
def sort_recover(fluctuated_value,revise_prediction_error):
    order = np.argsort(fluctuated_value)
    recover_prediction_error = np.zeros_like(revise_prediction_error)
    for idx,num in enumerate(revise_prediction_error):
        recover_prediction_error[order[idx]] = num
    # print(type(recover_prediction_error))
    return recover_prediction_error

def recover_black2x2(arr):
    r = np.zeros((512,512))
    flag = 0
    for i in range(1, 512-1):
        for j in range(1, 512 - 1):
            if (i + j) % 2 == 0:
                r[i][j] = arr[flag]
                flag += 1
    return r
def recover_white2x2(arr):
    r = np.zeros((512,512))
    flag = 0
    for i in range(1, 512-1):
        for j in range(1, 512 - 1):
            if (i + j) % 2 != 0:
                r[i][j] = arr[flag]
                flag += 1
    return r

def get_gray_pix(img):
    pix=[]
    width, hight = get_w_h(img)
    img=Image.open(img).convert("L")
    for i in range(hight):
        for j in range(width):
            pix.append(img.getpixel((j,i)))
    return pix


def generate_hist(data):
    min_err = int(min(data))
    max_err = int(max(data))
    bin = [i + 1 for i in list(range(min_err, max_err))]  # 设置条状的范围
    histogram, bins, patch = plt.hist(data, bin,facecolor='blue', histtype='stepfilled')  # histograming

    return histogram

def Find_min_Distancce(list,seek_num,flag_num,Critical_flag):#第一个是列表，第二是需要寻找的数，第三个是指定的参考数字,第四个为参考数字
    Distance=256#先预设一个最大的距离
    flag_min = 0
    for i in range(0,len(list)):
        if list[i]==seek_num:
            flag=i+1#这个是0点的下标
            if flag_num<Critical_flag:
                if abs(flag_num-flag)<Distance and (flag<flag_num or (flag>flag_num and flag<Critical_flag)):#比较得出最近的距离
                    Distance=abs(flag_num-flag)
                    flag_min = flag#保存一下当前距离的下标
            else:
                if abs(flag_num-flag)<Distance and (flag>flag_num or (flag<flag_num and flag>Critical_flag)):#比较得出最近的距离
                    Distance=abs(flag_num-flag)
                    flag_min = flag#保存一下当前距离的下标
    return flag_min

def max_and_min(prediction_error,show_histogram):

    #函数说明：用来将图片转换成直方图，并且返回直方图当中的像素数量最多的位置和次多的位置，以及返回离最高和次高位置的0点位置
    #形参数说明：pic_address图片地址   show_histogram为1显示直方图为0不显示
    #返回参数说明：MaxPix最多像素的个数,MaxPoint个数最多的像素值,To_MaxPoint_min_Point离个数最多的像素值最近的零点像素
    #           Second_MaxPix次多像素的个数,Second_MaxPoint个数次多的像素值,To_SecondMaxPoint_min_Point离个数次多的像素值最近的零点像素
    min_err = int(min(prediction_error))
    max_err = int(max(prediction_error))

    histogram = generate_hist(prediction_error)  # 产生直方图
    if show_histogram==1:
        plt.show()  # 展示直方图
    histogram_list = histogram.tolist()  # 转换成列表方便操作 这个列表里面存储的是0到255像素的个数
    # print(histogram_list)
    histogram_list_smalltobig = sorted(histogram_list)  # 对列表进行从小到大的排序
    MaxPix=histogram_list_smalltobig[-1]
    Second_MaxPix=histogram_list_smalltobig[-2]
    MaxPoint = histogram_list.index(MaxPix) + 1   # 获取最多数量像素的下标
    Second_MaxPoint = histogram_list.index(Second_MaxPix) + 1   # 获取第二多数量像素的下标
    # 下面寻找离最多个数下标最近的像素
    To_MaxPoint_min_Point = Find_min_Distancce(histogram_list, 0.0, MaxPoint,Second_MaxPoint)
    # 下面寻找离次多个数下标最近的像素
    To_SecondMaxPoint_min_Point = Find_min_Distancce(histogram_list, 0.0, Second_MaxPoint,MaxPoint)
    pk1pix, pk1, z1, pk2pix, pk2, z2 = 0,0,0,0,0,0
    #保持PK1在左   pk2在右边
    if Second_MaxPoint < MaxPoint :
        pk1 = Second_MaxPoint + min_err
        pk1pix = Second_MaxPix
        pk2 = MaxPoint + min_err
        pk2pix = MaxPix
        z1 = To_SecondMaxPoint_min_Point + min_err
        z2 = To_MaxPoint_min_Point + min_err
    elif Second_MaxPoint > MaxPoint:
        pk1 = MaxPoint + min_err
        pk1pix = MaxPix
        pk2 = Second_MaxPoint + min_err
        pk2pix = Second_MaxPix
        z1 = To_MaxPoint_min_Point + min_err
        z2 = To_SecondMaxPoint_min_Point + min_err

    return pk1pix,pk1,z1,pk2pix,pk2,z2
def SSIM(img1,img2):
# If the input is a multichannel (color) image, set multichannel=True.
#     print(ssim(img1, img2, multichannel=True))
    print(ssim(img1, img2))

def overflow(pixels):
    locationmap = np.zeros((len(pixels),len(pixels[0])))
    for i in range(len(pixels)):
        for j in range(len(pixels[0])):
            if pixels[i][j] == 0  :
                locationmap[i][j] = 1
                pixels[i][j] = 1
            elif pixels[i][j] == 255 :
                locationmap[i][j] = 1
                pixels[i][j] = 254
    return locationmap,pixels

def recoverflow(pixel,locationmap):
    for i in range(len(pixel)):
        for j in range(len(pixel[0])):
            if pixel[i][j] == 1 and locationmap[i][j] == 1  :
                pixel[i][j] = 0
            elif pixel[i][j] == 254 and locationmap[i][j] == 1 :
                pixel[i][j] = 255
    return pixel
