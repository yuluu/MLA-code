def file2Matrix(filename):
    arrayLines = open(filename).readlines()
    lenOfLines = len(arrayLines)
    featureMat = np.zeros((lenOfLines,3))
    classLabel = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split("\t")
        featureMat[index,:] = listFromLine[0:3]
        classLabel.append(int(listFromLine[-1]))
        index += 1
    return featureMat,classLabel

filename = "data/datingTestSet2.txt"
featureMat,classLabel = file2Matrix(filename)   
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(featureMat[:,0],featureMat[:,1],15.0*np.array(classLabel),15.0*np.array(classLabel))
plt.show()

def autoNorm(dataset):
    maxVals = np.max(dataset,axis=0)
    minVals = np.min(dataset,axis=0)
    normMat = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    normMat = dataset - np.tile(minVals,(m,1))
    normMat = normMat/np.tile(maxVals-minVals,(m,1))
    return normMat
    
def datingClassTest():
    testRatio = 0.1
    featureMat,classLabel = file2Matrix("data/datingTestSet2.txt")  
    normMat = autoNorm(featureMat)
    m = normMat.shape[0]   
    testNum = int(m*testRatio)
    errCount = 0
    for i in range(testNum):
        classifyResult = classify0(normMat[i,:],normMat[testNum:m,:],classLabel[testNum:m],3)
        if classifyResult != classLabel[i]:
            errCount += 1
    print("the total error rate is: %f" % (errCount/testNum))
datingClassTest()  
