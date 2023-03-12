# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 13:13:38 2023

@author: owner
"""

import numpy as np

class MLP(object):
    '''
    多層パーセプトロン
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        #入力層-隠れ層 
        self.l1 = Layer(input_dim=input_dim,
                        output_dim=hidden_dim,
                        activation=sigmoid,
                        dactivation=dsigmoid)
        #隠れ層-出力層 
        self.l2 = Layer(input_dim=hidden_dim,
                output_dim=output_dim,
                activation=sigmoid,
                dactivation=dsigmoid)
        
        self.layers = [self.l1, self.l2]
        
    def __call__(self, x):
        return self.forward(x)
        
    def forward(self, x):
        h = self.l1(x)
        y = self.l2(h)
        return y
    
    
class Layer(object):
    def __init__(self, input_dim, output_dim, activation, dactivation):
        '''
        インスタンス変数:
            W: 重み
            b: バイアス
            activation: 活性化関数
            dactivation: 活性化関数の微分
        '''
        
        self.W = np.random.normal(size=(input_dim, output_dim))
        self.b = np.zeros(output_dim)

        self.activation = activation
        self.dactivation = dactivation
    
    def __call__(self, x):
        return self.forward(x)
    
    #順伝播
    def forward(self, x):
        self._input = x
        self._pre_activation = np.matmul(x, self.W) + self.b
        return self.activation(self._pre_activation)
    #逆伝播
    def backward(self, delta, W):
        delta = self.dactivation(self._pre_activation) * np.matmul(delta, W.T)
        return delta
    #勾配計算
    def compute_gradients(self, delta):
        dW = np.matmul(self._input.T, delta)
        db = np.matmul(np.ones(self._input.shape[0]), delta)
        return dW, db
    
#シグモイド
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#シグモイドの微分
def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))