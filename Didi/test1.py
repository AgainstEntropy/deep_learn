import numpy as np
from pylab import *
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    # 定义数据集和标签
    dataMat = []
    x = [459.4, 454.5, 440.0]
    y = [142.4, 529.9, 1329.7]
    # 读取文件
    with open('test.txt') as f:
        # 计数器
        line_nu = 0
        # 一行行的把数据从硬盘加载到内存里读出来
        for line in f.readlines():
            # 读取前line_nu行
            if line_nu < 16000:
                # strip()把末尾的'\n'删掉 split分离数据
                lineArr = line.strip().split()
                dataMat.append([lineArr[2], lineArr[3]])
                line_nu += 1
            else:
                break
    print(line_nu)

    # 转为数组形式
    dataArr = array(dataMat)
    print(dataArr)

    # 画出所有点的信息
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 画散点图

    for i in range(line_nu):
                xcat = [dataArr[i, 0]]
                ycat = [dataArr[i, 1]]
                ax.scatter(xcat, ycat, c='r', marker='.')
    ax.scatter(x, y, c='b', marker='D')
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    plt.show()

if __name__ == '__main__':
    loadDataSet()
