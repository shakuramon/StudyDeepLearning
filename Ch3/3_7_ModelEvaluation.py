# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 02:15:12 2023

@author: owner
"""
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import matplotlib.pyplot as plt
from models import MLP

if __name__ == '__main__':
    np.random.seed(123)
    
    '''
    1.データの準備
    '''
    N = 300
    x, t = datasets.make_moons(N, noise = 0.3)

    x1 = np.zeros((150,2))
    x2 = np.zeros((150,2))
    i1 = 0
    i2 = 0
    for i in range(N):
        if t[i] == 0:
            x1[i1] = x[i]
            i1 += 1
        else:
            x2[i2] = x[i]
            i2 += 1
            
    plt.scatter(x1.T[0], x1.T[1])
    plt.scatter(x2.T[0], x2.T[1])
    plt.show()
    t = t.reshape(N, 1)
    print(t)
    
    #データの分割 訓練8:テスト2
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2)
    
    '''
    2.モデルの構築
    '''
    #model = MLP(2,2,1)
    model = MLP(2,3,1)
    
    '''
    3.モデルの学習
    '''
    def compute_loss(t, y):
        return (-t * np.log(y) - (1 - t) * np.log(1 - y)).sum()

    def train_step(x, t):
        y = model(x)
        for i, layer in enumerate(model.layers[::-1]):
            if i == 0:
                delta = y - t
            else:
                delta = layer.backward(delta, W)

            dW, db = layer.compute_gradients(delta)
            layer.W = layer.W - 0.1 * dW
            layer.b = layer.b - 0.1 * db

            W = layer.W

        loss = compute_loss(t, y)
        return loss

    epochs = 100
    batch_size = 30
    n_batches = x_train.shape[0] // batch_size

    for epoch in range(epochs):
        train_loss = 0.
        x_, t_ = shuffle(x_train, t_train)

        for n_batch in range(n_batches):
            start = n_batch * batch_size
            end = start + batch_size

            train_loss += train_step(x_[start:end],
                                     t_[start:end])

        if epoch % 10 == 0 or epoch == epochs - 1:
            print('epoch: {}, loss: {:.3f}'.format(
                epoch+1,
                train_loss
            ))

    '''
    4. モデルの評価
    '''
    preds = model(x_test) > 0.5
    acc = accuracy_score(t_test, preds)
    print('acc.: {:.3f}'.format(acc))
    
    precision = precision_score(t_test, preds)
    print('prec.: {:.3f}'.format(precision))
    
    recall = recall_score(t_test, preds)
    print('recall.: {:.3f}'.format(recall))
    
    f = 2 * precision * recall / (precision + recall)
    print('f-mean.: {:.3f}'.format(f))