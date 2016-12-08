# encoding: utf-8
from numpy import *

def loadDataSet(filename):
    dataMat=[]
    fr=open(filename)
    for line in fr.readlines():
        curline=line.strip().split("\t")
        fltline=map(float,curline)
        dataMat.append(fltline)
    return dataMat
def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))
def randCent(dataSet,k):
    n=shape(dataSet)[1]
    centroids=mat(zeros((k,n)))
    for j in range(n):
        minj=min(dataSet[:,j])
        rangej=float(max(dataSet[:,j])-minj)
        centroids[:,j]=minj+rangej*random.rand(k,1)
    return centroids
def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m=shape(dataSet)[0]
    clusterAssment=mat(zeros((m,2)))
    centroids=createCent(dataSet,k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        for i in range(m):
            minDist=inf;minIndex=-1
            for j in range(k):
                distJI=distMeas(centroids[j,:],dataSet[i,:])
                if distJI<minDist:
                    minDist=distJI
                    minIndex=j
            if clusterAssment[i,0]!=minIndex:
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2
    print centroids
   # for cent in range(k):



if __name__ == '__main__':
    d=mat(loadDataSet("/Users/ics/课程/机器学习/machinelearninginaction/Ch10/testSet.txt"))
    print min(d[:,0]),min(d[:,1]),max(d[:,0]),max(d[:,1])
    print randCent(d,2)
    print distEclud(d[0],d[1])