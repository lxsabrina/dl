
'''
@author: zhangxian
@brief Description : 交叉熵的python实现 ,X 2-d array , Y 1-d array

'''
import tensorflow as    tf
import numpy as np
import torch as torch

def cross_entropy_with_numpy(prob_y, y):    
   return -np.mean(np.sum(y * np.log(prob_y), axis = 1))

def cross_entropy_with_tensorflow(prob_y, y): #未测试
    return -tf.reduce_mean(tf.reduce_sum(y * tf.log(prob_y), axis = 1))

def cross_entropy_with_torch(prob_y, y):    #未测试 
    #x_np = torch.from_numpy(np_array) # ndarry -> tensor
    prob = torch.from_numpy(prob_y)
    gr = torch.from_numpy(y)
    return -torch.mean(torch.sum(gr * torch.log(prob), axis = 1))

def softmax_with_numpy(x):
    return np.exp(x)/np.sum(np.exp(x))  #分母这里是全部求和

def one_hot(x,cls):
    tmp = np.zeros((x.shape[0],cls))
    for i,k in enumerate(x):
        tmp[i][k] =1

def cross_entropy(X,Y):
    prob_y = softmax_with_numpy(X) 
    onehotY = one_hot(Y,2)
    return cross_entropy_with_numpy(prob_y, onehotY)


if __name__ == "__main__"::

    N  = 100000
    X = np.random.random([N,2])
    Y = (np.random.random([N,1])>0.5).astype(int)
    print(cross_entropy(X,Y))
