# -*- coding: utf-8 -*-

from assignment4.p1.k_means import *
dataSet = []
fileIn = open('realdata.txt')
for line in fileIn.readlines():  
    lineArr = line.strip().split('\t')  
    dataSet.append([float(lineArr[1]), float(lineArr[2])])
dataSet = mat(dataSet)
k = 2
centroids, clusterAssment = kmeans(dataSet, k)
showCluster(dataSet, k, clusterAssment)