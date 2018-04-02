
import numpy as np
from pylab import *
from numpy import *
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def loadDataSet():
    # 定义数据集和标签
    dataMat = []
    # 读取文件
    with open('test1.txt') as f:
        # 计数器
        line_nu = 0
        # 一行行的把数据从硬盘加载到内存里读出来
        for line in f.readlines():
            # 读取前line_nu行
            if line_nu < 50395+1:
                # strip()把末尾的'\n'删掉 split分离数据
                lineArr = line.strip().split(',')
                dataMat.append([lineArr[0], lineArr[2], lineArr[3]])
                line_nu += 1
            else:
                break
    print("一共"+str(line_nu)+"行")

    # 转为数组形式
    dataArr = array(dataMat)
    dotx= []
    doty= []

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

    xmajorLocator= MultipleLocator(100) #将x主刻度标签设置为20的倍数
    ymajorLocator= MultipleLocator(200) #将y轴主刻度标签设置为0.5的倍数

    #设置主刻度标签的位置
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    print("开始绘图")
    # 画散点图
    for i in range(line_nu-1):
        dotx = round(float(dataArr[i, 1]))
        doty = round(float(dataArr[i, 2]))
        print(i)
        ax.scatter(dotx, doty, c='r', marker='.')
    plt.show()


if __name__ == '__main__':
    loadDataSet()
