import numpy as np
from pylab import *
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def loadDataSet():
    # 定义数据集和标签
    dataMat = []
    # 读取文件
    with open('test.txt') as f:
        # 计数器
        line_nu = 0
        # 一行行的把数据从硬盘加载到内存里读出来
        for line in f.readlines():
            # 读取前line_nu行
            if line_nu < 799111:
                # strip()把末尾的'\n'删掉 split分离数据
                lineArr = line.strip().split()
                dataMat.append([lineArr[0], lineArr[2], lineArr[3]])
                line_nu += 1
            else:
                break
    print(line_nu)

    # 创建一个二维数组
    lists = [[] for i in range(16111)]

    # 转为数组形式
    dataArr = array(dataMat)
    # 把str数据改为浮点数后用round函数四舍五入为整数
    #print(round(float(dataArr[0, 1])))

    k = 1
    for i in range(1, line_nu):
        #print(i)
        if(int(dataArr[i-1, 0]) == k):
            #print(k)
            #lists[k-1].append([dataArr[i-1, 1], dataArr[i-1, 2]])
            dotx = round(float(dataArr[i-1, 1]))
            doty = round(float(dataArr[i-1, 2]))
            lists[k-1].append([dotx, doty])
            k += 1
        else:
            k = int(dataArr[i-1, 0])
            #lists[k-1].append([dataArr[i-1, 1], dataArr[i-1, 2]])
            dotx = round(float(dataArr[i-1, 1]))
            doty = round(float(dataArr[i-1, 2]))
            lists[k-1].append([dotx, doty])
            k += 1
    #print(lists)

    # 画出所有点的信息
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')

    # 提高MAXTICKS值
    locator = mdates.MinuteLocator(byminute=[0, 30])
    locator.MAXTICKS = 1500
    ax.xaxis.set_major_locator(locator)

    xmajorLocator= MultipleLocator(10) #将x主刻度标签设置为20的倍数
    ymajorLocator= MultipleLocator(10) #将y轴主刻度标签设置为0.5的倍数

    #设置主刻度标签的位置
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)

    # 画散点图
    for i in range(16111):
                xcat = [lists[i][0][0]]
                ycat = [lists[i][0][1]]
                ax.scatter(xcat, ycat, c='r', marker='.')
    for i in range(16111):
                xcat = [lists[i][-1][0]]
                ycat = [lists[i][-1][1]]
                ax.scatter(xcat, ycat, c='b', marker='.')
    plt.show()


if __name__ == '__main__':
    loadDataSet()
