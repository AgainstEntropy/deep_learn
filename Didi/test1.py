import numpy as np
from pylab import *
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def loadDataSet():
    # 定义数据集和标签
    dataMat = []
    x =[ 521677, 521580, 521520, 521452, 521433, 521411, 521400]
    y =[ 58109, 57466, 57059, 56668, 55855, 54822, 53998]
    # 读取文件
    with open('test1.txt') as f:
        # 计数器
        line_nu = 0
        # 一行行的把数据从硬盘加载到内存里读出来
        for line in f.readlines():
            # 读取前line_nu行   609395+1
            if line_nu < 609395+1:
                # strip()把末尾的'\n'删掉 split分离数据
                lineArr = line.strip().split(',')
                dataMat.append([lineArr[0], lineArr[2], lineArr[3]])
                line_nu += 1
            else:
                break
    print("一共"+str(line_nu)+"行")

    # 转为数组形式
    dataArr = array(dataMat)

    j = 0
    for i in range(line_nu-1):
        if(dataArr[i, 0] != dataArr[i+1, 0]):
            j += 1
    print("id一共有"+str(j)+"个")

    # 创建一个二维数组
    lists = [[] for i in range(j+1)]
    # 把str数据改为浮点数后用round函数四舍五入为整数
    #print(round(float(dataArr[0, 1])))

    k = 0
    for i in range(line_nu-1):
        dotx = round(float(dataArr[i, 1]))
        doty = round(float(dataArr[i, 2]))
        lists[k].append([dotx, doty])
        if(dataArr[i, 0] != dataArr[i+1, 0]):
            k += 1

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
    # 画散点图
    for i in range(j):
                xcat = [lists[i][0][0]]
                ycat = [lists[i][0][1]]
                ax.scatter(xcat, ycat, c='r', marker='.')
    for i in range(j):
                xcat = [lists[i][-1][0]]
                ycat = [lists[i][-1][1]]
                ax.scatter(xcat, ycat, c='b', marker='.')
    ax.scatter(x, y, c='y', marker='x')
    plt.show()


if __name__ == '__main__':
    loadDataSet()
