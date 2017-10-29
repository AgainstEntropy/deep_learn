from numpy import *
# 导入运算符模块
import operator
from os import listdir

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# 核心算法
def classify0(inX,dataSet,labels,k):
        # 训练数据集的行数
        dataSetSize = dataSet.shape[0]
        # 计算距离
        # 这里要说一下tile()函数，以后我们还会多次用到它
        # tile(A,B)表示对A重复B次，B可以是int型也可以是数组形式
        # 如果B是int，表示在行方向上重复A，B次，列方向默认为1
        # 如果B是数组形式，tile(A,(B1,B2))表示在行方向上重复B1次，列方向重复B2次
        diffMat = tile(inX, (dataSetSize, 1))-dataSet
        print(diffMat)
        sqDiffMat = diffMat**2
        print(sqDiffMat)
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances**0.5
        # 排序，这里argsort()返回的是数据从小到大的索引值,这里这就是第几行数据
        sortedDisIndicies = distances.argsort()
        print(sortedDisIndicies)
        classCount={}
        # 选取距离最小的k个点，并统计每个类别出现的频率
        # 这里用到了字典get(key,default=None)返回键值key对应的值；
        # 如果key没有在字典里，则返回default参数的值，默认为None
        for i in range(k):
                voteIlabel = labels[sortedDisIndicies[i]]
                classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
        # 逆序排序，找出出现频率最多的类别
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        print(sortedClassCount)
        return sortedClassCount[0][0]

# 读取txt数据的代码
def file2matrix(filename):
        fr = open(filename)
        # 读取文件
        arrayOLines = fr.readlines()
        # 文件行数
        numberOfLines = len(arrayOLines)
        # 创建全0矩阵
        returnMat = zeros((numberOfLines, 3))
        # 标签向量
        classLabelVector = []
        index = 0
        # 遍历每一行，提取数据
        for line in arrayOLines:
                line = line.strip()
                listFromLine = line.split('\t')
                # 前三列为属性信息
                returnMat[index, :] = listFromLine[0:3]
                # 最后一列为标签信息
                classLabelVector.append(int(listFromLine[-1]))
                index += 1
        return returnMat, classLabelVector

# 归一化特征值
def autoNorm(dataSet):
        # 每列的最小值
        minVals = dataSet.min(0)
        # 每列的最大值
        maxVals = dataSet.max(0)
        # 最大值与最小值的差值
        ranges = maxVals-minVals
        normDataSet = zeros(shape(dataSet))
        m = dataSet.shape[0]
        # minVals是1*3的矩阵，使用tile函数复制成和dataSet同样大小的矩阵，方便计算
        normDataSet = dataSet-tile(minVals, (m, 1))
        normDataSet = normDataSet/tile(ranges, (m, 1))
        return normDataSet, ranges, minVals

# 原始测试分类器
def datingClassTest():
        hoRatio = 0.10
        datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
        normMat, ranges, minVals = autoNorm(datingDataMat)
        m = normMat.shape(0)
        # 10%的数据用于测试数据集
        numTestVecs = int(m*hoRatio)
        errorCount = 0.0
        for i in range(numTestVecs):
                classifierResults = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
                print("the classifier came back with: %d,the real answer id: %d" % (classifierResults, datingLabels[i]))
                if(classifierResults!=datingLabels[i]):errorCount += 1.0
        print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

# 约会网站预测函数
def classifyPerson():
        resultList = ['not at all', 'in small doses', 'in large doses']
        percentTats = float(input("在游戏上花费的时间占比( )?"))
        ffMiles = float(input("每年航空的里程数?"))
        iceCream = float(input("每年吃的冰淇淋（升）?"))
        datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
        normMat, ranges, minVals = autoNorm(datingDataMat)
        inArr = array([ffMiles, percentTats, iceCream])
        classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
        print("你可能是属于以下这类人：", resultList[classifierResult - 1])

# 手写测试
# 将图片格式处理为一个向量（把一个32*32的二进制图形矩阵转换为1*1024的向量）,然后使用分类器处理图形信息
def img2vector(filename):
    # 创建一个1*1024的NumPy数组
    returnVect = zeros((1, 1024))
    fr = open(filename)
    # 循环读出文件的前32行，并将每行的头32个字符值存储在NumPy数组中，最后返回数组
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    # 将trainingDigits目录中的文件存储在列表中
    trainingFileList = listdir('trainingDigits')
    # 获得文件数赋值给m
    m = len(trainingFileList)
    # 创建一个m行1024列的训练矩阵，每行数据存储一个图像
    trainingMat = zeros((m, 1024))
    for i in range(m):
        # 从文件名解析分类数字（eg：文件9_45.txt表示数字9的第45个实例）
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        # 将类代码存储在 hwLabels向量中
        hwLabels.append(classNumStr)
        # 使用 img2vector函数载入图像
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    # 将testDigits目录中的文件存储在列表中，步骤同上
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        # 不同的是要使用 classify0函数测试目录下的每个文件（文件中的值介于0-1之间，无需用autoNorm()函数
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))

if __name__ == '__main__':
    # classifyPerson()
    handwritingClassTest()
    # datingClassTest()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), array(datingLabels))
    # plt.show()