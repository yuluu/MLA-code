import numpy as np
import operator

def createDateset():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ["A","A","B","B"]
    return group,labels

def classify0(test,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    #tile() 函数，就是将原矩阵横向、纵向地复制
    diffMat = np.tile(test,(dataSetSize,1)) - dataSet #测试样本到每个训练样本的坐标距离
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    print(distances)
    sortedDistances = distances.argsort() # argsort返回的是数组值从小到大的索引值
    print(sortedDistances)
    classCount = {}
    for i in range(k):
        if labels[sortedDistances[i]] not in classCount:
            classCount[labels[sortedDistances[i]]] = 1
        else:
            classCount[labels[sortedDistances[i]]] += 1
    sortedClassCount = sorted(classCount.items(),key=lambda item:item[1],reverse=True)
    return sortedClassCount[0][0]

group,labels = createDateset()
classify0([0,0],group,labels,3)
