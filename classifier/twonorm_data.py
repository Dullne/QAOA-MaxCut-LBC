import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.model_selection import train_test_split,cross_val_score, KFold,ShuffleSplit
from core.Hiq_class import *
import time 
import os
#7400*20 class(2)
def plt_data():
    file_path = 'datasets/twonorm.dat'
    twonorm = pd.read_csv(file_path, header=None,  delimiter=', ',skiprows=25)
    y=twonorm[20]
    X=twonorm.drop(20,axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(2)  # project from 64 to 2 dimensions
    X= pca.fit_transform(X)

    # tsne = TSNE(n_components=2, random_state=42)
    # X = tsne.fit_transform(X)

    return X,y
def batch(X_test,y_test,i):
    X_batch=X_test[10*i:10*(i+1)]
    y_batch=y_test[10*i:10*(i+1)]
    return X_batch,y_batch
def split_data(X,y,test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42,shuffle=True)
    X_train_0,X_train_1=X_train[y_train==0],X_train[y_train==1]
    y_test=np.array(y_test)

    centroid_0=np.mean(X_train_0,axis=0)
    centroid_1=np.mean(X_train_1,axis=0)

    return centroid_0,centroid_1,X_test,y_test
def model():
    time_start=time.time()
    batch_n=148
    test_size=0.2
    X,y=plt_data()
    centroid_0,centroid_1,X_test,y_test=split_data(X,y,test_size)

    result=0
    p_2d=[]
    loss_2d=[]

    for i in range(batch_n):
        print(f"Fold {i}:")
        X_batch,y_batch=batch(X_test,y_test,i)
        len_X_batch=len(X_batch)

        X_batch=np.insert(X_batch,0,list(centroid_0),axis=0)
        y_batch=np.insert(y_batch,0,0)
        X_batch=np.insert(X_batch,1,list(centroid_1),axis=0)
        y_batch=np.insert(y_batch,1,1)

        p,losses,steps=classifer(data_list=X_batch,y=y_batch)
        result+=p*len_X_batch
    acc=result/len(X_test)


    print("Accuracy:",acc)
    time_span=time.time()-time_start
    print('run time:',time_span)

    return acc,p_2d,loss_2d,dataset_name

if __name__=="__main__":
    acc,p_2d,loss_2d,dataset_name=model() 
