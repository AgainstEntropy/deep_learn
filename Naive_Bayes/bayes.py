from numpy import *

# 产生训练数据
def loadDataSet():
    # 该数据取自某狗狗论坛的留言版
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 标注每条数据的分类，这里0表示正常言论，1表示侮辱性留言
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

# 建立词汇表
def createVocabList(dataSet):
    # 首先建立一个空集
    vocabSet = set([])
    # 遍历数据集中的每条数据
    for document in dataSet:
        # 这条语句中首先统计了每条数据的词汇集，然后与总的词汇表求并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 按照词汇表解析输入
def setOfWords2Vec(vocabList, inputSet):
    # 创建一个跟词汇表（vocabList）等长的向量，并将其元素都设为0
    returnVec = [0]*len(vocabList)
    # 遍历输入，将含有词汇表单词的文档向量设为1
    for word in inputSet:
        if word in vocabList:
            # 现在每遇到一个单词会增加词向量中的对应量
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word:%s is not in my vocabulary!" % word)
    return returnVec

# 朴素贝叶斯分类器训练函数
# 输入参数trainMatrix表示输入的文档矩阵，trainCategory表示每篇文档类别标签所构成的向量
def trainNB0(trainMatrix,trainCategory):
    # 留言数目
    numTrainDocs=len(trainMatrix)
    # 变换矩阵的列数目，即词汇表数目
    numWords=len(trainMatrix[0])
    # 侮辱性留言的概率
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=ones(numWords)
    p1Num=ones(numWords)
    # 将所有词的出现数初始化为1，将分母初始化为2，从而降低计算多个概率的乘积结果为零的影响
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        # 统计每类单词的数目，注意我们这里讨论的是一个二分问题
        # 所以可以直接用一个if...else...即可，如果分类较多，则需要更改代码
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 对每个类别除以该类中的总词数
    # 防止下溢出
    p1Vec = log(p1Num/p1Denom)
    p0Vec = log(p0Num/p0Denom)
    # 函数返回两个概率向量，及一个概率
    return p0Vec, p1Vec, pAbusive

# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass):
    p1 = sum(vec2Classify*p1Vec)+log(pClass)
    p0 = sum(vec2Classify*p0Vec)+log(1-pClass)
    if p1 > p0:
        return 1
    else:
        return 0

#内嵌测试函数
def testingNB():
    listOPosts, listClasses=loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
      trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, p1 = trainNB0(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print(testEntry, "classified as:", classifyNB(thisDoc, p0V, p1V, p1))
    testEntry = ['garbage', 'stupid']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print(testEntry, "classified as:", classifyNB(thisDoc, p0V, p1V, p1))

# 使用朴素贝叶斯进行垃圾邮件过滤
# 该函数将每个句子都解析成单词，并忽略空格，标点符号以及长度小于3的单词
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# 检测垃圾邮件
def spamTest():
    # 存放输入数据
    docList = []
    #存放类别标签
    classList = []
    # 所有的文本
    fullText = []
    # 分别读取邮件内容
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i, "rb").read().decode('GBK','ignore') )
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i,  "rb").read().decode('GBK','ignore') )
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    # range(50)表示从0到50，不包括50
    trainingSet = list(range(50))
    # 测试集
    testSet = []
    # 随机抽取是个作为测试集
    for i in range(10):
        # 从50个数据集中随机选取十个作为测试集，并把其从训练集中删除
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []

    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    # 使用训练集得到概率向量
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))

    # 测试分类器的错误率
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("Classification error:")
            print(docList[docIndex])
    print(errorCount)
    print("the error rate is:", float(errorCount)/len(testSet))

if __name__ == '__main__':
    #testingNB()
    spamTest()