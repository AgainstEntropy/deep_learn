from numpy import *
import matplotlib.pyplot as plt
def loadDataSet():
    # 定义数据集和标签
    dataMat = []
    labelMat = []
    # 读取文件
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 初始化数据
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

# 回归函数
def sigmoid(intX):
    return 1.0/(1+exp(-intX))

# 梯度上升算法
def gradAscent(dataMatIn,classLabels):
    # 转换为Numpy数据类型
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    # 矩阵大小
    m, n = shape(dataMatrix)
    # 步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    # 系数矩阵初始化为1
    weights = ones((n, 1))
    for k in range(maxCycles):
        # 变量h是一个列向量，元素个数等于样本个数
        # 变量h和误差error都是向量
        h = sigmoid(dataMatrix*weights)
        error = (labelMat-h)
        weights = weights+alpha*dataMatrix.transpose()*error
    return weights

# 随机梯度上升算法
def stocGradAscent0(dataMatrix,classLabels):
    # 无矩阵转换过程
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        # 变量h和误差error都是数值
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = (classLabels[i]-h)
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # alpha在每次迭代时不断减小，但不会减到0
            alpha = 4/(1.0+j+i)+0.0001
            # 随机选取更新(减少周期波动)
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 删除，进行下一次迭代
            del(dataIndex[randIndex])
    return weights

# 画出最佳拟合直线
def plotBestFit(weights):
    '''
    # 矩阵变为数组,使用gradAscent时加入
    weights = wei.getA()
    '''
    # 加载数据
    dataMat, labelMat = loadDataSet()
    # 转化为数组
    dataArr = array(dataMat)
    # 数据的列数目
    n = shape(dataArr)[0]
    # 用于存放类1的点
    xcord1 = []
    ycord1 = []
    # 用于存放类2的点
    xcord2 = []
    ycord2 = []
    # 遍历所有点
    for i in range(n):
        if(int(labelMat[i]) == 1):
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    # 画出所有点的信息
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    # 画出分类的边界，函数的系数由之前的梯度上升算法求得
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X1')
    plt.show()

# 通过输入回归系数和特征向量来计算对应sigmoid的值
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    # 导入数据
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 导入数据完成后利用stocGradAscent1（）来计算回归系数向量
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    # 导入测试集并计算分类错误率
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

# 调用colicTest()函数10次并求结果的平均值
def multiTest():
    numTests = 10
    errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))


if __name__ == '__main__':
    '''
    dataArr, labelMat = loadDataSet()
    weight = gradAscent(dataArr, labelMat)
    weight1 = stocGradAscent0(array(dataArr), labelMat)
    weight2 = stocGradAscent0(array(dataArr), labelMat)
    plotBestFit(weight2)
    '''
    multiTest()