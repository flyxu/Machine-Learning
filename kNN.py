# encoding: utf-8

from numpy import *
import operator

def classify(inx,dataset,labels,k):
    datasetsize=dataset.shape[0]
    diffmat=tile(inx,(datasetsize,1))-dataset
    print diffmat
    sqdiffmat=diffmat**2   #每个元素的平方
    print sqdiffmat
    sqdistances=sqdiffmat.sum(axis=1) #计算每个行向量的和
    print sqdistances
    distance=sqdistances**0.5
    sortedDistIndicies=distance.argsort() #返回从小到大的序值
    print sortedDistIndicies
    classCount={}
    for i in range(k):
        votelabel=labels[sortedDistIndicies[i]]
        classCount[votelabel]=classCount.get(votelabel,0)+1  #如果标签不存在返回1
    #print classCount
    #print classCount.iteritems()
    sortedClassCount=sorted(classCount.iteritems(),
    key=operator.itemgetter(1),reverse=True)
    print type(sortedClassCount[0])
    #print sortedClassCount[0][0]
    return sortedClassCount[0][0]
def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
if __name__=="__main__":
    group,labels=createDataSet()
    label=classify([0,0],group,labels,3)
    print label

