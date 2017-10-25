#import numpy
import numpy as np
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
y = np.array([[0, 0, 1, 1]]).T

# 为随机数设定产生的种子
# deterministic (just a good practice)
np.random.seed(1)
# 将随机初始化的权重矩阵均值设定为 0
# syn0(第一层网络间的权重矩阵)
syn0 = 2*np.random.random((3,1)) - 1

# sigmoid 函数
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

for iter in range(10000):
    # forward propagation
    l0 = X
    # 计算输入层的加权和，即用输入矩阵L0乘以权重矩阵syn0，并通过sigmid函数进行归一化。得到输出结果l1
    l1 = nonlin(np.dot(l0,syn0))

    # 计算输出结果L1与真实结果y之间的误差L1_error
    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the
    # 计算权重矩阵的修正L1_delta，即用误差乘以sigmoid在L处的导数
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    # 用L1_delta更新权重矩阵syn0
    syn0 += np.dot(l0.T,l1_delta)

# 输出迭代的结果
print(syn0)

# 加入新的测试已验证
X_new = np.array([[0,1,0],
                  [1,0,0]])
y_new = np.dot(X_new,syn0)
print(y_new)
