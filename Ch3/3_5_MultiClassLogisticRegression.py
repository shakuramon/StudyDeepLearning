# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 15:30:58 2023

@author: owner
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class LogisticRegression(object):
    '''
    多クラスロジスティック回帰
    '''
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.w_mat = np.random.normal(size=(input_dim, output_dim))
        self.b_vec = np.zeros(output_dim)
        
    def __call__(self, x_vec):
        return self.forward(x_vec)
    
    def forward(self, x_vec):
        return softmax(np.matmul(x_vec, self.w_mat) + self.b_vec)
    
    def compute_gradients(self, x_vec, t_mat):
        y = self.forward(x_vec)
        delta = y - t_mat
        dw = np.matmul(x_vec.T, delta)
        db = np.matmul(np.ones(x_vec.shape[0]), delta)
        return dw, db
        

def softmax(x_vec):
    return np.exp(x_vec) / np.sum(np.exp(x_vec), axis=1, keepdims=True)
            

if __name__ == '__main__':
    np.random.seed(123)
    
    '''
    1.データの準備
    '''
    M = 2       #入力データの次元
    K = 3       #クラス数
    n = 100     #クラスごとのデータ数
    N = n * K   #全データ数
    
    x1 = np.random.randn(n, M) + np.array([0,10])#0,10中心の点群
    x2 = np.random.randn(n, M) + np.array([5,5])#5,5中心の点群
    x3 = np.random.randn(n, M) + np.array([10,0])#10,10中心の点群
    
    t1 = np.array([[1, 0, 0] for i in range(n)])
    t2 = np.array([[0, 1, 0] for i in range(n)])
    t3 = np.array([[0, 0, 1] for i in range(n)])
    
    x = np.concatenate((x1, x2, x3), axis=0)#配列の結合
    t = np.concatenate((t1, t2, t3), axis=0)#配列の結合
    
    plt.scatter(x1.T[0], x1.T[1])
    plt.scatter(x2.T[0], x2.T[1])
    plt.scatter(x3.T[0], x3.T[1])
    plt.show()    
    
    '''
    2.モデルの構築
    '''
    model = LogisticRegression(input_dim=M, output_dim=K)
    
    '''
    3.モデルの学習
    '''
    def compute_loss(t, y):
        return (-t * np.log(y)).sum(axis=1).mean()#平均
    
    def train_step(x_mat, t_mat):#勾配
        dw, db = model.compute_gradients(x_mat, t_mat)
        model.w_mat = model.w_mat - 0.1 * dw
        model.b_vec = model.b_vec - 0.1 * db
        loss = compute_loss(t_mat, model(x_mat))
        return loss
    
    epochs = 10
    batch_size = 50
    n_batches = x.shape[0] // batch_size
    
    for epoch in range(epochs):
        train_loss = 0.
        x_, t_ = shuffle(x, t)#シャッフル
        for n_batch in range(n_batches):
            start = n_batch * batch_size#ミニバッチ適応
            end = start + batch_size
            
            train_loss += train_step(x_[start:end], t_[start:end])#勾配降下法
            
        if epoch % 10 == 0 or epoch == epochs - 1:
            print('epoch: {}, loss: {:.3f}'.format(epoch+1, train_loss))
    
    '''
    4.モデルの評価
    '''
    x_, t_ = shuffle(x, t)
    preds = model(x_[0:5])
    classified = \
        np.argmax(t_[0:5], axis = 1) == np.argmax(preds[0:5], axis=1)
    print('Prediction matched:', classified)
    print(preds)
    print(model.w_mat)
    print(model.b_vec)
    
    '''
    グラフ描画
    '''
    mat = model.w_mat.T
    x = np.arange(-1,5)
    y = ((mat[1][0] - mat[0][0]) * x + (model.b_vec[1] - model.b_vec[0]))/(mat[0][1] - mat[1][1])
    xx = np.arange(-4,13)
    yy = ((mat[2][0] - mat[1][0]) * xx + (model.b_vec[2] - model.b_vec[1]))/(mat[1][1] - mat[2][1])
    plt.scatter(x1.T[0], x1.T[1])
    plt.scatter(x2.T[0], x2.T[1])
    plt.scatter(x3.T[0], x3.T[1])
    plt.plot(x, y)
    plt.plot(xx,yy)
    plt.show()
    
    
    
    
    
    