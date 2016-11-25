import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def readArff(fileName):
    arffFile = open(fileName,'r')
    data = []
    for line in arffFile.readlines():
        if not (line.startswith('@')):
            if not (line.startswith('%')):
                if line !='\n':
                    L=line.strip('\n')
                    k=L.split(',')
                    data.append(k)
    return data

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  # convert to NumPy matrix
    labelMat = np.mat(classLabels).transpose()  # convert to NumPy matrix
    m, n = np.shape(dataMatrix)
    maxCycles = 500
    weights = np.ones((n, 1))
    listw=[]
    lista=[]
    j=-2
    for i in range(30):
         lista.append(j)
         j+=0.2
    for alpha in lista:
        for k in range(maxCycles):  # heavy on matrix operations
            h = sigmoid(dataMatrix * weights)  # matrix mult
            error = (labelMat - h)  # vector subtraction
            weights = weights + alpha * dataMatrix.transpose() * error  # matrix mult
        listw.append(weights)
    return listw,lista

def test_LogRegres(listw, test_x, test_y):
    numSamples, numFeatures = np.shape(test_x)
    tp,fp,fn= 0,0,0
    listfm=[]
    for weights in listw:
        for i in range(numSamples):
            predict = sigmoid(test_x[i] * weights)[0, 0] > 0.5
            if(predict == bool(test_y[i]) and predict == True):
                tp += 1
            elif(predict != bool(test_y[i]) and bool(test_y[i])== True):
                fn+=1
            elif(predict != bool(test_y[i]) and bool(test_y[i])== False):
                fp+=1
        pre=tp/(tp+fp)
        rec=tp/(tp+fn)
        f_measure =(2*pre*rec) / (pre+rec)
        listfm.append(f_measure)
    return listfm

def test_LogRegres_s(listw, test_x, test_y):
    numSamples, numFeatures = np.shape(test_x)
    tp,fp,fn= 0,0,0
    listfm=[]
    for weights in listw:
        for i in range(numSamples):
            predict = sigmoid(test_x[i] * weights)[0, 0] > 0.5
            if(predict == bool(test_y[i])== True):
                tp += 1
            elif(predict != bool(test_y[i]) and bool(test_y[i])== True):
                fn+=1
            elif(predict != bool(test_y[i]) and bool(test_y[i])== False):
                fp+=1
        pre=tp/(tp+fp)
        rec=tp/(tp+fn)
        f_measure =1-(2*pre*rec) / (pre+rec)
        listfm.append(f_measure)
    return listfm

def GetResult():
    data = readArff('chronic_kidney_disease_full.arff')
    label=[]
    for temp in data:
        if (len(temp) != 25):
            temp.remove('')
        if(temp[24]=='ckd'):
            temp[24]=float(1)
        else:
            temp[24]=float(0)
        label.append(temp[24])
        del temp[24]
    for temp in data:
        for i in range(24):
            if (temp[i] == '?' or temp[i] == '\t?'):
                temp[i] = float(0)
            elif(temp[i]=='normal' or temp[i]=='present' or temp[i]=='yes'or temp[i]=='good' or temp[i]==' yes' or temp[i]=='\tyes'):
                temp[i]=float(2)
            elif(temp[i]=='abnormal' or temp[i]=='notpresent' or temp[i]=='no' or temp[i]=='poor' or temp[i]=='\tno'):
                temp[i]=float(1)
            else:
                temp[i]=float(temp[i])

    docs_train, docs_test, y_train, y_test = train_test_split(
        data, label, test_size=0.2, random_state=None)

    listw,lista= gradAscent(docs_train, y_train)
    f_measure=test_LogRegres(listw, docs_test, y_test)
    plt.figure(1)
    plt.plot(lista, f_measure, 'g')
    plt.xlabel("lambda")
    plt.ylabel("f_measure")
    plt.show()

    meantrain=np.mean(np.array(docs_train),axis= 0)
    meantest = np.mean(np.array(docs_test), axis=0)
    stdtrain=np.std(np.array(docs_train),axis= 0)
    stdtest = np.std(np.array(docs_test), axis=0)
    docs_train=(np.array(docs_train)-meantrain)/stdtrain
    docs_test = (np.array(docs_test) - meantest)/stdtest
    listw, lista = gradAscent(docs_train.tolist(), y_train)
    f_measure = test_LogRegres_s(listw, docs_test.tolist(), y_test)
    plt.figure(2)
    plt.xlabel("lambda")
    plt.plot(lista, f_measure, 'r')
    plt.ylabel("f_measure with standardization")
    plt.show()

if __name__ == '__main__':
    GetResult()