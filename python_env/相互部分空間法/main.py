# %%
import numpy as np
from tqdm import tqdm 
import Option
import util
import Convert
from Coef_update_l1 import Coef_update_L1
from Dict_update_L1 import Dict_update_L1
import Image_IO
import Encode 
from Decode_Fourier_L2 import Decode_Fourier_L2
from Decode_Random_L2 import Decode_Random_L2
from sporco import plot 
import matplotlib.pyplot as plt
import intial as N
from data import load_fashionMNIST
from base import base
from functools import partial
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Sequential
from sporco import plot
from sporco import util as ut


N.TRAIN=4
N.TEST=10
train_data, train_target, test_data, test_target = load_fashionMNIST()
train_data=train_data.transpose(0,3,1,2).reshape(10,N.TRAIN,28,28)
test_data=test_data.transpose(2,0,1).reshape(N.TEST,28,28)

train_target=train_target.astype(int).reshape(10*N.TRAIN)

#ITER=15,N.FILTER=18,FILTER_SIZE=19,LAMBDA=0.0007,RHO=27
#%%
for P in range(30,120,200):
    ITER=P

    for Q in range(18,31,200):
        N.FILTER[0]=Q

        for C in range(19,29,300):
            N.FILTER_SIZE[0]=C
            N.FILTER_SIZE[1]=C

            for A in range(1,700,6009):
                LAMBDA=0.0007

                for B in range(1,10,100):
                    RHO=27

                    #訓練画像に対しての学習
                    D_TRAIN=np.zeros((10,N.TRAIN,N.FILTER[0],N.FILTER_SIZE[0],N.FILTER_SIZE[0]))
                    
                    for j in tqdm(range(10)):
                        for m in range(N.TRAIN):

                            np.random.seed(seed=123)
                            dictlearn_opt = Option.DictLearn_Option(N=28*28, M=N.FILTER[0], Filter_Width=N.FILTER_SIZE[0], Lambda=LAMBDA, Rho=RHO, coef_iteration=2, dict_iteration=2)
                            D = np.random.normal(0, 1, (dictlearn_opt.M, dictlearn_opt.Filter_Width, dictlearn_opt.Filter_Width))
                            D = util.padding(D, dictlearn_opt)
                            X = np.zeros((dictlearn_opt.M, dictlearn_opt.N))
                            KF=train_data[j,m].flatten()
                            Coef = Coef_update_L1(dictlearn_opt, KF)
                            Dict = Dict_update_L1(dictlearn_opt, KF) 
                            Dict.D = D
                            for i in range(ITER):
                                X = Coef.coef_update_l1(D)
                                D = Dict.dict_update_l1(X)
                            #Dict.get_tiled_D() 
                            D_TRAIN[j,m]=D.reshape((dictlearn_opt.M, 28, 28))[:,:dictlearn_opt.Filter_Width,:dictlearn_opt.Filter_Width]

                    #テスト画像に対しての学習
                    D_TEST=np.zeros((N.TEST,N.FILTER[0],N.FILTER_SIZE[0],N.FILTER_SIZE[0]))
                    np.random.seed(seed=123)
                    dictlearn_opt = Option.DictLearn_Option(N=28*28, M=N.FILTER[0], Filter_Width=N.FILTER_SIZE[0], Lambda=LAMBDA, Rho=RHO, coef_iteration=2, dict_iteration=2)
                    
                    for j in tqdm(range(N.TEST)):
                        D = np.random.normal(0, 1, (dictlearn_opt.M, dictlearn_opt.Filter_Width, dictlearn_opt.Filter_Width))
                        D = util.padding(D, dictlearn_opt)
                        X = np.zeros((dictlearn_opt.M, dictlearn_opt.N))
                        KF=test_data[j].flatten()
                        Coef = Coef_update_L1(dictlearn_opt, KF)
                        Dict = Dict_update_L1(dictlearn_opt, KF) 
                        Dict.D = D
                        #Dict.get_tiled_D() 
                
                        for i in range(ITER):
                            X = Coef.coef_update_l1(D)
                            D = Dict.dict_update_l1(X)

                        D_TEST[j]=D.reshape((dictlearn_opt.M, 28, 28))[:,:dictlearn_opt.Filter_Width,:dictlearn_opt.Filter_Width]
                        #print(D_TEST[j].shape)


                    D_TRAIN_1=D_TRAIN.reshape(10,N.TRAIN,N.FILTER[0],-1).transpose(0,1,3,2)
                    D_TEST_1=D_TEST.reshape(N.TEST,N.FILTER[0],-1).transpose(0,2,1)
                    
                    #%%各画像の辞書フィルタを元に主成分分析　基底を求める。
                    B_TRAIN=np.zeros([10,N.TRAIN,N.FILTER_SIZE[0]*N.FILTER_SIZE[1],N.FILTER_SIZE[0]*N.FILTER_SIZE[1]])
                    for i in range(10):
                        for j in range(N.TRAIN):
                            B_TRAIN[i,j,:,:]=base(D_TRAIN_1[i,j,:,:])

                    B_TEST=np.zeros([N.TEST,N.FILTER_SIZE[0]*N.FILTER_SIZE[1],N.FILTER_SIZE[0]*N.FILTER_SIZE[1]])
                    for i in range(N.TEST):
                        B_TEST[i,:,:]=base(D_TEST_1[i,:,:])
                    


                    #%%相互部分空間法
                    for train in range(1,N.TRAIN+1,1):
                        
                        Accuracy=0
                        Accuracy2=0
                        Accuracy3=0

                        #学習画像　基底の数
                        for i in range(1,50,3):
                            #訓練画像　基底の数
                            for l in range(1,50,3):
                                accuracy=0
                                accuracy2=0
                                accuracy3=0
                                #各テスト画像と全学習画像とを相互部分空間法で分類。
                                for j in range(N.TEST):
                                    out = np.empty(10)
                                    out2=np.zeros([10])
                                    LA=np.zeros([10,train])
                                    out3=np.zeros([10])
                                    
                                    for k in range(10):
                                        for m in range(train):
                                            A=np.dot(B_TRAIN[k,m,:,:i].T,B_TEST[j,:,:l])
                                            B=np.dot(A,B_TEST[j,:,:l].T)
                                            lam, v = np.linalg.eigh(np.dot(B,B_TRAIN[k,m,:,:i]))  
                                            LA[k,m]=max(lam) 
                                        LA[k,:]=np.sort(LA[k,:])[::-1]
                                        out[k]=LA[k,0]
                                        out2[k]=sum(LA[k,:])
                                        if train > 2:
                                            for w in range(3):
                                                out3[k]+=LA[k,w]
                                    if np.argmax(out) == test_target[j]:
                                        accuracy+= 1  
                                    if np.argmax(out2) == test_target[j]:
                                        accuracy2+= 1  
                                    if np.argmax(out3) == test_target[j]:
                                        accuracy3+= 1  
                                if Accuracy<accuracy:
                                    Accuracy=accuracy
                                if Accuracy2<accuracy2:
                                    Accuracy2=accuracy2 
                                if Accuracy3<accuracy3:
                                    Accuracy3=accuracy3  

                        print(Accuracy/N.TEST,Accuracy2/N.TEST,Accuracy3/N.TEST,LAMBDA,RHO,N.FILTER[0],N.FILTER_SIZE[0],train,N.TEST,ITER)

# %%
