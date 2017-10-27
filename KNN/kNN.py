from numpy import *
# 导入运算符模块
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX,dataSet,labels,k):
        #训练数据集的行数
        dataSetSize=dataSet.shape[0]
        #计算距离
        #这里要说一下tile()函数，以后我们还会多次用到它
        #tile(A,B)表示对A重复B次，B可以是int型也可以是数组形式
        #如果B是int，表示在行方向上重复A，B次，列方向默认为1
        #如果B是数组形式，tile(A,(B1,B2))表示在行方向上重复B1次，列方向重复B2次
        diffMat=tile(inX,(dataSetSize,1))-dataSet
        print(diffMat)
        sqDiffMat=diffMat**2
        print(sqDiffMat)
        sqDistances=sqDiffMat.sum(axis=1)
        distances=sqDistances**0.5
        #排序，这里argsort()返回的是数据从小到大的索引值,这里这就是第几行数据
        sortedDisIndicies =distances.argsort()
        print(sortedDisIndicies)
        classCount={}
        #选取距离最小的k个点，并统计每个类别出现的频率
        #这里用到了字典get(key,default=None)返回键值key对应的值；
        #如果key没有在字典里，则返回default参数的值，默认为None
        for i in range(k):
                voteIlabel=labels[sortedDisIndicies[i]]
                classCount[voteIlabel]=classCount.get(voteIlabel,0)+1;
        #逆序排序，找出出现频率最多的类别
        sortedClassCount=sorted(classCount.iteritems(),
                                key=operator.itemgetter(1),reverse=True)
        print(sortedClassCount)
        return sortedClassCount[0][0]

# 读取txt数据的代码
def file2matrix(filename):
        fr=open(filename)
        #读取文件
        arrayOLines=fr.readlines()
        #文件行数
        numberOfLines=len(arrayOLines)
        #创建全0矩阵
        returnMat=zeros((numberOfLines,3))
        #标签向量
        classLabelVector=[]
        index=0
        #遍历每一行，提取数据
        for line in arrayOLines:
                line=line.strip();
                listFromLine=line.split('\t')
                #前三列为属性信息
                returnMat[index,:]=listFromLine[0:3]
                #最后一列为标签信息
                classLabelVector.append(int(listFromLine[-1]))
                index +=1
        return returnMat,classLabelVector
