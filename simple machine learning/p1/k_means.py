# -*- coding: utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt

def euclDistance(vector1, vector2):
    return sqrt(sum(power(vector2 - vector1, 2)))  

def initCentriods(dataSet,k):
    #print(dataSet)
    numSamples,dim = dataSet.shape
    centroids = zeros((k, dim))    
    print("row：",numSamples,",","column：",dim)
    for i in range(k):
        index = int(random.uniform(0, numSamples)) 
        centroids[i, :] = dataSet[index, :]
    return centroids
# k-means cluster  
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    clusterAssment = mat(zeros((numSamples, 2)))
    clusterChanged = True
    centroids = initCentriods(dataSet, k)
    #print("initial centroids：",centroids)

    while clusterChanged: 
        clusterChanged = False
        for i in range(numSamples):
            minDist  = 100000.0 
            minIndex = 0
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist: 
                    minDist  = distance
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2
        for j in range(k):  
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]] 
            centroids[j, :] = mean(pointsInCluster, axis = 0) 
    print('done!')
    return centroids, clusterAssment

def showCluster(dataSet, k, clusterAssment):
    numSamples, dim = dataSet.shape
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
    if k > len(mark):  
        print("Sorry! Your k is too large!")
        return 1  
  
    # draw all samples  
    for i in range(numSamples):  
        markIndex = int(clusterAssment[i, 0])  
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  

    plt.xlabel("Length")
    plt.ylabel("Width")
    plt.show()