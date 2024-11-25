import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.model_selection import train_test_split,cross_val_score, KFold,ShuffleSplit
from core.Hiq_class import *
import matplotlib.pyplot as plt
import time 
#215*5 class(3)   150/35/30
def make_data():
    file_path = 'datasets/newthyroid.dat'
    newthyroid = pd.read_csv(file_path, header=None, delimiter=',', skiprows=10)
    indices = np.where((newthyroid[5] == 2) | (newthyroid[5] == 3))[0]
    newthyroid=newthyroid.iloc[indices,:]
    y=np.array(newthyroid[5].map({2:0,3:1}))
    X=newthyroid.drop(5,axis=1)
   
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # pca = PCA(2)
    # X = pca.fit_transform(X)
    # tsne = TSNE(n_components=2, random_state=42)
    # X = tsne.fit_transform(X)

    return X,y
def split_data(X,y,n_split,test_size):
    ss = ShuffleSplit(n_splits=n_split, test_size=test_size, random_state=42)
    result=0
    for i,(train_idx, test_idx) in enumerate(ss.split(X)):
        print(f"Fold {i+1}:")
        X_train, X_test,y_train,y_test = X[train_idx], X[test_idx],y[train_idx],y[test_idx]
        class_indices = {}
        for class_label in np.unique(y_train):
            class_indices[class_label] = np.where(y_train == class_label)[0]
        centroids = {}
        for class_label, indices in class_indices.items():
            class_data = X_train[indices]
            centroids[class_label] = np.mean(class_data, axis=0)
        data_list=X_test.tolist()
        y_list=y_test.tolist()
        for class_label, centroid in centroids.items():
            data_list=np.insert(data_list,class_label,list(centroid),axis=0)
            y_list=np.insert(y_list,class_label,class_label)
            
        p,losses,steps=classifer(data_list=data_list,y=y_list) 
        result+=p*(len(X_test))

    acc=result/len(X)
    return acc

if __name__=="__main__":
    time_start = time.time()
    n_split=5
    test_size=0.2
    X,y=make_data()
    acc=split_data(X,y,n_split,test_size)
    print("Accuracy=",acc)
    time_span = time.time() - time_start
    print('run time:', time_span)

