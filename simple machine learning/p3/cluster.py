from sklearn import datasets
from assignment4.p1.k_means import *
from sklearn import metrics
from sklearn import cluster
import numpy as np

digits = datasets.load_digits()
dataSet = mat(digits.data)
numSamples, dim = dataSet.shape
k = 10

def result(clusterAssment):
    c1,c2,c3,c4,c5,c6,c7,c8,c9,c0=[],[],[],[],[],[],[],[],[],[]
    for i in range(numSamples):
        if(int(clusterAssment[i])==1):
            c1.append(digits.target[i])
        elif(int(clusterAssment[i])== 2):
            c2.append(digits.target[i])
        elif (int(clusterAssment[i]) == 3):
            c3.append(digits.target[i])
        elif (int(clusterAssment[i]) == 4):
            c4.append(digits.target[i])
        elif (int(clusterAssment[i]) == 5):
            c5.append(digits.target[i])
        elif (int(clusterAssment[i]) == 6):
            c6.append(digits.target[i])
        elif (int(clusterAssment[i]) == 7):
            c7.append(digits.target[i])
        elif (int(clusterAssment[i]) == 8):
            c8.append(digits.target[i])
        elif (int(clusterAssment[i]) == 9):
            c9.append(digits.target[i])
        elif (int(clusterAssment[i]) == 0):
            c0.append(digits.target[i])

    s0,s1,s2,s3,s4,s5,s6,s7,s8,s9=[],[],[],[],[],[],[],[],[],[]
    for i in range(10):
        s0.append(c0.count(i))
    for i in range(10):
        s1.append(c1.count(i))
    for i in range(10):
        s2.append(c2.count(i))
    for i in range(10):
        s3.append(c3.count(i))
    for i in range(10):
        s4.append(c4.count(i))
    for i in range(10):
        s5.append(c5.count(i))
    for i in range(10):
        s6.append(c6.count(i))
    for i in range(10):
        s7.append(c7.count(i))
    for i in range(10):
        s8.append(c8.count(i))
    for i in range(10):
        s9.append(c9.count(i))

    label=[]
    label.append(s0.index(max(s0)))
    label.append(s1.index(max(s1)))
    label.append(s2.index(max(s2)))
    label.append(s3.index(max(s3)))
    label.append(s4.index(max(s4)))
    label.append(s5.index(max(s5)))
    label.append(s6.index(max(s6)))
    label.append(s7.index(max(s7)))
    label.append(s8.index(max(s8)))
    label.append(s9.index(max(s9)))
    print("the result of cluster: ")
    print(label)

    pred=[]
    for i in range(numSamples):
        for j in range(10):
            if (int(clusterAssment[i]) == j):
                pred.append(label[j])
    cm = metrics.confusion_matrix(digits.target, pred)
    print("the confusion matrix is below: ")
    print(cm)

    cx=np.sum(cm,axis=0)
    cy=np.sum(cm,axis=1)
    tp,fp,fn,FM=[],[],[],[]
    for j in range(10):
        tp.append(cm[j,j])
        fp.append(cx[j]-cm[j,j])
        fn.append(cy[j]-cm[j,j])
        FM.append(((cm[j,j]/cx[j])*(cm[j,j]/cy[j]))**0.5)
    F_M=0
    for temp in FM:
        F_M=float(F_M+temp)
    print("The Fowlkes–Mallows index is: ",float(F_M/10))

centroids, clusterAssment = kmeans(dataSet, k)
pred=[]
for i in range(numSamples):
    pred.append(clusterAssment[i,0])
print("Result information about my kmeans:")
result(pred)

ward = cluster.AgglomerativeClustering(n_clusters=k, linkage='ward')
pred=ward.fit_predict(digits.data)
print("Result information about Agglomerative clustering with Ward linkage:")
result(pred)

affinity_propagation = cluster.AffinityPropagation()
pred=affinity_propagation.fit_predict(digits.data)
print("The result of cluster fot AffinityPropagation is below:")
print(pred)