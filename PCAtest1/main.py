import numpy as np #矩阵运算库
import matplotlib.pyplot as plt #图形显示
'''
该算法为PCA算法，主程序分析算法
本次算法调用数据：diabetes.csv 鸢尾花的各项数据
前四列为数据，最后一列为标签
'''

def readingDatas():
    '''
    读入数据
    :return:data_array 类别是 numpy.genfromtxt
    '''
    data_array = np.genfromtxt("./diabetes.csv",delimiter=',',skip_header=False)
    lableList = data_array.T[-1]
    data_array = np.delete(data_array, -1, axis=1) #s删除最后一列
    return data_array,lableList

def zeroMean(dataSet):
    '''
    将数据矩阵减去每一个特征的均值
    :param dataSet:类别是 numpy.genfromtxt，是数据集
    :return: newData类别是 numpy.genfromtxt
    '''
    meanVal = np.mean(dataSet, axis=0)  # 按列求均值，即求各个特征的均值
    newData = dataSet - meanVal #二维矩阵减去一维矩阵
    return newData


def covarianceMatrix(dataSet):
    '''
    求出协方差矩阵，为n*n的矩阵，n为特征维度
    :param dataSet:输入为数据矩阵
    :return:covariance:返回协方差为矩阵
    '''
    # covariance = np.dot(dataSet.T , dataSet) #没有除以m
    covariance = np.cov(dataSet.T.astype(float))
    #print(covariance)
    return covariance


def eigenValuesVectors(covariance,k):
    '''
        计算对应的特征值和特征向量,并将特征值按照从大到小排序
    :param covariance: 协方差矩阵
    :param k: 取前k行
    :return:
    '''
    vals,vecs = np.linalg.eigh(covariance)
    print("特征值：\n")
    print(vals)
    print("特征向量：\n")
    print(vecs)
    eigen = zip(vals,vecs)
    eigen = sorted(eigen,reverse=True)
    #for i in eigen:
    #     print(i)
    eigen = list(zip(*eigen))
    print(eigen)
    vals = eigen[0]
    vecs = eigen[1]
    basisVector =vecs[:k]
    basisVector = np.array(basisVector, dtype=float)
    #print(basisVector)
    return basisVector


def pca(dataSet,k):
    '''
    PCA算法，
    :param dataSet:
    :param k:
    :return:
    '''

    covariance = covarianceMatrix(zeroMean(dataSet))
    basisVector = eigenValuesVectors(covariance, k)
    data = np.dot(basisVector,dataSet.T).T
    #print(data)
    return data

def pltshow3(data,labelList):
    '''
    降维后数据展示，展示降维到三维时的状态
    :return:
    '''

    colors = labelList.astype(np.str)
    colors[colors == '1.0'] = 'r'
    colors[colors == '2.0'] = 'b'
    colors[colors == '3.0'] = 'g'
    #print(colors)
    fig = plt.figure()
    ax1 = fig.add_subplot(111,projection='3d')
    ax1.set_title('3D Plot')
    ax1.scatter(data.T[0], data.T[1], data.T[2], c=colors, marker='o')
    plt.show()



def pltshow2(data, labelList):
    '''
    降维后数据展示，展示降维到二维时的状态
    :return:
    '''
    # dataList = list(zip(data, lableList))

    colors = labelList.astype(np.str)
    colors[colors == '1.0'] = 'r'
    colors[colors == '2.0'] = 'b'
    colors[colors == '3.0'] = 'g'
    #print(colors)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Scatter Plot')
    ax1.scatter(data.T[0],data.T[1], c=colors, marker='o')
    plt.show()

def origin(data,labelList):
    '''
    展示原数据图
    :param data:
    :param labelList:
    :return:
    '''
    colors = labelList.astype(np.str)
    colors[colors == '1.0'] = 'r'
    colors[colors == '2.0'] = 'b'
    colors[colors == '3.0'] = 'g'
    # print(colors)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Original Plot')
    ax1.scatter(data.T[0], data.T[1], c=colors, marker='o')
    plt.show()

def origin3d(data, labelList):
    '''
    展示原数据图
    :param data:
    :param labelList:
    :return:
    '''
    colors = labelList.astype(np.str)
    colors[colors == '1.0'] = 'r'
    colors[colors == '2.0'] = 'b'
    colors[colors == '3.0'] = 'g'
    # print(colors)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection = '3d')
    ax1.set_title('Original3d Plot')
    ax1.scatter(data.T[0], data.T[1], data.T[2], c=colors, marker='o')
    plt.show()

def main():
    '''
    测试PCA算法，
    '''
    dataSet, labelList = readingDatas()
    print(dataSet)

    print('pca算法原始二维图：\n')
    origin(dataSet,labelList)

    data2 = pca(dataSet, 2)
    print('PCA算法降维到二维：\n', data2)
    pltshow2(data2, labelList)

    print('pca算法原始三维图：\n')
    origin3d(dataSet,labelList)
    data3 = pca(dataSet, 3)
    print('PCA算法降维到二维：\n', data3)
    pltshow3(data3, labelList)


if __name__ == "__main__":
    print("PCA算法：")
    main()


