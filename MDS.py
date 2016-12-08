#encoding:utf-8

import numpy as np
sonar_train_PATH='./DataSet/sonar-train.txt'
sonar_test_PATH='./DataSet/sonar-test.txt'
splice_train_PATH='./DataSet/splice-train.txt'
splice_test_PATH='./DataSet/splice-test.txt'
def loadfile(filepath):
    f=open(filepath)
    data=[]
    for line in f.readlines():
        line=line.strip().split(',')
        fltline=map(float,line)
        data.append(fltline)
    return np.array(data)

def KNN(x,train,label,k=1):
    diff=np.tile(x,(len(train),1))-train
    diff=diff**2
    distance=diff.sum(axis=1)**0.5
    dis_index=distance.argsort()
    dis_k_index=dis_index[0:k]
    label_k=label[dis_k_index]
    dict={}
    for lab in label_k:
        if lab in dict.keys():
            dict[lab]=dict[lab]+1
        else:
            dict[lab]=0
    sorted_dict=sorted(dict.iteritems(),key=lambda x:x[1],reverse=True)
    return sorted_dict[0][0]

def KNN_predict(test,train,label,k=1):
    predict_label=[]
    for x in test:
        predict=KNN(x,train,label,k)
        predict_label.append(predict)
    return np.array(predict_label)


def get_distance(vector):
    return np.sqrt(sum(vector**2))

def dis_matrix(train):
    m=len(train)
    distance=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            distance[i][j]=get_distance(train[i]-train[j])
            #print distance[i][j]
    return distance

def get_B(train):
    distance=dis_matrix(train)
    m=len(train)
    B=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            B[i][j]=-0.5*(distance[i][j]**2-sum(distance[i])**2-sum(distance[:,j])**2+distance.sum()**2)
            #print B[i][j]
    return B

def mds(train,n):
    B=get_B(train)
    eigVals, eigVects = np.linalg.eig(np.mat(B))
    sort_index=eigVals.argsort()
    sort_n_index=sort_index[:-(n+1):-1]
    eigVals_k=eigVals[sort_n_index]
    eigVals_diag=np.diag(eigVals_k)**0.5
    eigVects_k=eigVects[:,sort_n_index]
    return np.array(eigVects_k*eigVals_diag)


def get_accuracy(test,train,train_label,test_label,k=1):
    N = [10, 20, 30]
    accuracy_list=[]
    for n in N:
        pca_train=mds(train,n)
        pca_test=mds(test,n)
        predict_label=KNN_predict(pca_test,pca_train,train_label,k=1)
        count=0
        for i in range(len(test_label)):
            if predict_label[i]==test_label[i]:
                count+=1
        accuracy = float(count) / len(test_label)
        accuracy_list.append(accuracy)
    return np.array(accuracy_list)




if __name__=='__main__':
    sonar_train = loadfile(sonar_train_PATH)
    sonar_test = loadfile(sonar_test_PATH)
    splice_train = loadfile(splice_train_PATH)
    splice_test = loadfile(splice_test_PATH)
    #sonar_accuracy = get_accuracy(sonar_test[:, 0:-1], sonar_train[:, 0:-1], sonar_train[:, -1], sonar_test[:, -1], k=1)
    print'---------------------------------------------------'
    print 'sonar数据集在mds处理后维度分别为：10，20，30'
    N = [10, 20, 30]
    # for i in range(len(sonar_accuracy)):
    #     print '维度为%s,对应的准确率为%s' % (N[i], sonar_accuracy[i])
    splice_accuracy = get_accuracy(splice_test[:, 0:-1], splice_train[:, 0:-1], splice_train[:, -1], splice_test[:, -1],k=1)
    print'---------------------------------------------------'
    print 'splice数据集在mds处理后维度分别为：10，20，30'
    for i in range(len(splice_accuracy)):
        print '维度为%s,对应的准确率为%s' % (N[i], splice_accuracy[i])
    print'---------------------------------------------------'