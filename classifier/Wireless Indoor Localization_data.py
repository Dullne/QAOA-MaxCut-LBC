import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.model_selection import train_test_split,cross_val_score, KFold,ShuffleSplit
from Hiq_class import *
import time 
from Hiq_class_acc import *

def plt_data():
    # 指定bupa.dat文件路径
    file_path = 'D:\paper\线性可分感知机\wireless+indoor+localization\wifi_localization.txt'
    # 读取数据集
    wireless = pd.read_csv(file_path, header=None,  delimiter='\t',skiprows=0)
    # wireless.to_csv('wireless.csv')
    print(wireless.shape)
    # ionosphere = ionosphere.rename(columns={33:"class"})
    # print(wireless["class"].unique())
    # wireless[7]=wireless[7].map({'g':0,'b':1})
    y=wireless[7]
    X=wireless.drop(7,axis=1)
    # sonar["class"]=sonar["class"].str.strip().map({'R':0,'M':1}).astype('int')
    # print(sonar.shape)
    # sonar.to_csv('sonar.csv')
    # # 数据预处理
    # y = sonar['class']
    # X = sonar.drop('class', axis=1)
   
    # print(y,y.unique())
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # ion=pd.concat([pd.DataFrame(X),y],axis=1)
    # print(ion)
    # ion.to_csv('ionosphere.csv')
    # print(X.shape)
    # print(y.shape)

    # tsne = TSNE(n_components=2, random_state=42)
    # X_2d = tsne.fit_transform(X)

    u = PCA(2)
    X_2d = u.fit_transform(X)
    # 绘制结果
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1,2,1)
    # for i in range(1,5):
    #     plt.scatter(X_2d[y==i,0], X_2d[y==i, 1],alpha=0.5,label=str(i))
    # # plt.xlabel('component 1')
    # # plt.ylabel('component 2')
    # plt.title('t-SNE Visualization of MNIST Data (2D)')
    # plt.legend()

    # pca = PCA(2)  # project from 64 to 2 dimensions
    # X_2d = pca.fit_transform(X)
    # plt.subplot(1,2,2)
    # for i in range(1,5):
    #     plt.scatter(X_2d[y==i,0], X_2d[y==i, 1],alpha=0.5,label=str(i))
    # plt.title('PCA Visualization of MNIST Data (2D)')
    # plt.legend()
    # plt.show()
    return X_2d,y
def make_data():
    file_path = 'D:\paper\线性可分感知机\wireless+indoor+localization\wifi_localization.txt'
    # 读取数据集
    wireless = pd.read_csv(file_path, header=None, delimiter='\t', skiprows=0)
    # ionosphere.to_csv('ionosphere.csv')
    # print(seeds)
    # ionosphere = ionosphere.rename(columns={33:"class"})
    # print(ionosphere["class"].unique())
    # pima[4]=pima[4].map({'tested_positive':0,'tested_negative':1})
    indices = np.where((wireless[7] == 1) | (wireless[7] == 2))[0]
    # print(indices)
    wireless=wireless.iloc[indices,:]
    y=np.array(wireless[7].map({1:0,2:1}))
    X=wireless.drop(7,axis=1)
    # sonar["class"]=sonar["class"].str.strip().map({'R':0,'M':1}).astype('int')
    # print(sonar.shape)
    # sonar.to_csv('sonar.csv')
    # # 数据预处理
    # y = sonar['class']
    # X = sonar.drop('class', axis=1)
   
    # print(banknote)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(X.shape)
    print(y.shape)


    # pca = PCA(2)  # project from 64 to 2 dimensions
    # X_2d = pca.fit_transform(X)
    # # 绘制结果
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1,2,1)
    # for i in range(2):
    #     plt.scatter(X_2d[y==i,0], X_2d[y==i, 1],alpha=0.5,label=str(i))
    # plt.title('PCA Visualization of MNIST Data (2D)')
    # plt.legend()

    # tsne = TSNE(n_components=2, random_state=42)
    # X_2d = tsne.fit_transform(X)
    
    # fig=plt.figure(figsize=(8,7))
    # plt.rcParams.update({'font.size':20})
    # plt.scatter(X_2d[y==0,0], X_2d[y==0, 1],alpha=0.5,label='class1')
    # plt.scatter(X_2d[y==1,0], X_2d[y==1, 1],alpha=0.5,label='class2')
    # plt.xlabel('component 1')
    # plt.ylabel('component 2')
    # plt.title('Wireless Indoor Localization')
    # plt.legend()

    # fig.tight_layout(rect=[0, 0, 1, 1])
    # plt.savefig('二维数据分布图/wireless.pdf')
    # plt.show()
    # print(X_2d,y)
    
    # return X_2d,y
    return X,y


def batch(X_test,y_test,i):
    X_batch=X_test[10*i:10*(i+1)]
    y_batch=y_test[10*i:10*(i+1)]
    return X_batch,y_batch
def split_data(X,y,test_size):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42,shuffle=True)
    X_train_0,X_train_1=X_train[y_train==0],X_train[y_train==1]
    y_test=np.array(y_test)
    # print(X_test.shape,y_test.shape)
    print(X_train_0.shape,X_train_1.shape)
    # pd.DataFrame(X_train_1).to_csv('banknote_X_train_1.csv')
    centroid_0=np.mean(X_train_0,axis=0)
    centroid_1=np.mean(X_train_1,axis=0)
    # print(y_test.shape)
    return centroid_0,centroid_1,X_test,y_test
if __name__=="__main__":
    time_start=time.time()
    batch_n=20
    test_size=0.2
    X,y=make_data()
    
    centroid_0,centroid_1,X_test,y_test=split_data(X,y,test_size)
    # # print(centroid_0,centroid_1)
    result=0
    # fig1=plt.figure(figsize=(8,6))
    # fig2=plt.figure(figsize=(8,6))
    # plt.rcParams.update({'font.size':20})
    for i in range(batch_n):
        print(f"Fold {i}:")
        X_batch,y_batch=batch(X_test,y_test,i)
        len_X_batch=len(X_batch)
        # print(X,type(y),y)
        X_batch=np.insert(X_batch,0,list(centroid_0),axis=0)
        y_batch=np.insert(y_batch,0,0)
        X_batch=np.insert(X_batch,1,list(centroid_1),axis=0)
        y_batch=np.insert(y_batch,1,1)
        # print(X,type(y),y)
        p,losses,steps=classifer(data_list=X_batch,y=y_batch)
    #     # # if i%2==0:
    #     # #     plt.plot(steps,losses,label=f'Batch {i+1}')
        result+=p*len_X_batch
    #     # print(len_X_batch)

    #     p_list,losses,steps=classifer_accfig(data_list=X_batch,y=y_batch)#画ACC图，对应Hiq_class_acc.py
    #     if i%2==0:
    #         plt.figure(1)
    #         plt.plot(steps,p_list,label=f'Batch {i + 1}') #ACC图
    #         plt.figure(2)
    #         plt.plot(steps,losses,label=f'Batch {i + 1}')
    #     result+=p_list[-1]*len_X_batch

    # fig1.tight_layout(rect=[0.02, 0.02, 1, 0.98])
    # fig2.tight_layout(rect=[0.02, 0.02, 1, 0.98])
    # dataset_name = "Wireless Indoor Localization"
    # plt.figure(1)
    # plt.xlabel("Batch iterations")
    # plt.ylabel('Accuracy')
    # # plt.figtext(0.5, 0.01, f"Dataset: {dataset_name}", ha="center", fontsize=10, color="black")
    # plt.title(f'{dataset_name}')
    # plt.legend()
    # plt.savefig(f'Acc图/{dataset_name}.pdf')

    # plt.figure(2)
    # plt.xlabel("Batch iterations")
    # plt.ylabel('Loss')
    # plt.title(f'{dataset_name}')
    # plt.legend()
    # plt.savefig(f'Loss损失图/{dataset_name}.pdf')

    # # plt.xlabel("Iterations")
    # # plt.ylabel('Loss')
    # # plt.legend()
    # # dataset_name = "Wireless Indoor Localization"
    # # plt.title(f'{dataset_name}')
    # # # plt.savefig('Wireless.pdf')
    # # plt.show()

    
    # # plt.xlabel("Iterations")
    # # plt.ylabel('Acc')
    # # # plt.figtext(0.5, 0.01, f"Dataset: {dataset_name}", ha="center", fontsize=10, color="black")
    # # plt.title(f'{dataset_name}')
    # # plt.legend()
    # # plt.savefig('Acc图/Wireless.pdf')
    # # plt.show()
    acc=result/len(X_test)
    print(X_test.shape)
    print("acc为：",acc)

    time_span=time.time()-time_start
    print('主程序段总共运行了',time_span,'秒')