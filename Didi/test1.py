import numpy as np
from pylab import *
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    # 定义数据集和标签
    dataMat = []
    xcat = []
    ycat = []
    x = [459.4, 454.5, 440.0]
    y = [142.4, 529.9, 1329.7]
    # 读取文件
    with open('test.txt') as f:
        # 计数器
        line_nu = 0
        # 一行行的把数据从硬盘加载到内存里读出来
        for line in f.readlines():
            # 读取前line_nu行
            if line_nu < 1000:
                # strip()把末尾的'\n'删掉 split分离数据
                lineArr = line.strip().split()
                dataMat.append([lineArr[0], lineArr[2], lineArr[3]])
                line_nu += 1
            else:
                break
    print(line_nu)

    dataArr = array(dataMat)
    print(dataArr)

    # 画出所有点的信息
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #设置x轴范围
    xlim(0, 1000)
    #设置y轴范围
    ylim(-500, 2000)
    # 画散点图
    ax.scatter(x, y, c='b', marker='D')
    k = 0
    for i in range(1, line_nu):
        if(0<=float(dataArr[i-1, 1])<1000 & -500<=float(dataArr[i-1, 2])<2000):
                xcat = [dataArr[i-1, 1]]
                ycat = [dataArr[i-1, 2]]
                ax.scatter(xcat, ycat, c='r', marker='.')
                k += 1
                if(dataArr[i, 1]!=dataArr[i-1, 1]):
                    plt.pause(0.1)
    print(k)
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    plt.show()

if __name__ == '__main__':
    loadDataSet()
