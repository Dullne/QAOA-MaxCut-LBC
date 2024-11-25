import pandas as pd
import numpy as np
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,normalize,Normalizer
from core.Hiq_class import *
from core import mnist_reader
import os

def data():
    print(os.getcwd())
    X_train, y_train = mnist_reader.load_mnist('datasets/MNIST', kind='train')
    X_test, y_test = mnist_reader.load_mnist('datasets/MNIST', kind='t10k')
    X_train,X_test=pd.DataFrame(X_train/255.0),pd.DataFrame(X_test/255.0)
    y_train,y_test=pd.DataFrame(y_train,columns=["target"]),pd.DataFrame(y_test,columns=["target"])

    X_train_69=X_train[(y_train['target']==6)|(y_train['target']==9)]
    y_train_69=y_train[(y_train['target']==6)|(y_train['target']==9)]
    X_test_69=X_test[(y_test['target']==6)|(y_test['target']==9)]
    y_test_69=y_test[(y_test['target']==6)|(y_test['target']==9)]


    tsne = TSNE(2)  
    X_train_69_2d = pd.DataFrame(tsne.fit_transform(X_train_69), index=X_train_69.index)
    X_test_69_2d=pd.DataFrame(tsne.fit_transform(X_test_69), index=X_test_69.index)

    # pca = PCA(2)  
    # X_train_69_2d = pd.DataFrame(pca.fit_transform(X_train_69), index=X_train_69.index)
    # X_test_69_2d=pd.DataFrame(pca.fit_transform(X_test_69), index=X_test_69.index)

    
    y_test_69=y_test_69['target'].map({9:0,6:1})
    X_test_69_2d=np.array(X_test_69_2d)
    y_test_69=np.array(y_test_69)

    X_train_6_2d=X_train_69_2d[(y_train_69['target']==6)]
    X_train_9_2d=X_train_69_2d[(y_train_69['target']==9)]

    centroid_6=np.array(X_train_6_2d.mean())
    centroid_9=np.array(X_train_9_2d.mean())

    centroid_6=normalize(centroid_6.reshape(1,-1))
    centroid_9=normalize(centroid_9.reshape(1,-1))
    X_test_69_2d=normalize(X_test_69_2d)

    return centroid_6,centroid_9,X_test_69_2d,y_test_69

def batch(X_69_2d_test,y_69_test,i):
    X=X_69_2d_test[10*i:10*(i+1),:]
    y=y_69_test[10*i:10*(i+1)]
    return X,y
    
if __name__=="__main__":
    time_start=time.time()

    #mnist
    centroid_6_train,centroid_9_train,X_69_2d_test,y_69_test=data()
    batch_n=197

    result=0
    for i in range(batch_n):
        print(f'{i+1} batch:')
        X,y=batch(X_69_2d_test,y_69_test,i)
        len_X=len(X)
        X=np.insert(X,0,list(centroid_9_train),axis=0)
        y=np.insert(y,0,0)
        X=np.insert(X,1,list(centroid_6_train),axis=0)
        y=np.insert(y,1,1)
        p,losses,steps=classifer(data_list=X,y=y)
        result+=p*len_X
    
    acc=result/len(X_69_2d_test)
    print("accuracy:",acc)
    time_span=time.time()-time_start
    print(time_span)



