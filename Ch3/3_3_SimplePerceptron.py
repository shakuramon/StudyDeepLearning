# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt


class SimplePerceprton(object):
    '''
    単純パーセプトロン
    '''
    #イニシャライズ
    def __init__(self, input_dim):
        self.input_dim = input_dim
        #wは重みベクトル（0で初期化すると上手くいかない時があるらしい）
        self.w = np.random.normal(size=(input_dim,))
        #bはバイアス
        self.b = 0.
        
    #ステップ関数の結果
    def forward(self, inputTbl):
        y = step(np.matmul(self.w, inputTbl) + self.b)
        return y
        
    #学習部分(誤差計算)
    def compute_deltas(self, x, t):
        y = self.forward(x)
        delta = y - t
        dw = delta * x
        db = delta
        return dw, db
        
#パーセプトロンが発火するかどうか
def step(x):
    return 1 * (x >= 0)
    
if __name__ == '__main__':
    np.random.seed(123) #乱数シード
    
    '''
    データ準備
    '''
    d = 2 # 入力次元
    N = 20 # 全データ
    
    mean = 5
    
    x1 = np.random.randn(N//2, d) + np.array([0, 0])
    x2 = np.random.randn(N//2, d) + np.array([mean, mean])
    
    t1 = np.zeros(N//2)
    t2 = np.ones(N//2)
    
    x = np.concatenate((x1, x2), axis=0) # 入力データ
    t = np.concatenate((t1, t2))         #出力データ
    
    # グラフ作成
    plt.scatter(x1.T[0], x1.T[1])
    plt.scatter(x2.T[0], x2.T[1])
    plt.show()
    
    plt.scatter(t1.T[0], t1.T[1])
    plt.scatter(t2.T[0], t2.T[1])
    plt.show()

    
    '''
    モデル構築
    '''
    model = SimplePerceprton(input_dim=d)
    
    a = 1
    b = a + 1
    print(b)
    
    
    '''
    モデルの学習
    '''
    def compute_loss(dw, db):
        print(dw)
        print(dw == 0)
        return all(dw == 0) * (db == 0)
    
    def train_step(x, t):
        # 与えられたデータを用いてパラメータを更新
        dw, db = model.compute_deltas(x, t)
        loss = compute_loss(dw, db)
        model.w = model.w - dw
        model.b = model.b - db
        return loss
    # -------------------------------------------------------
    while True:
        classified =True
        for i in range(N):
            loss = train_step(x[i], t[i])
            classified *= loss
        if classified:
            break
        '''
        else:
            print('false')
        '''
            
    '''
    モデル評価
    '''
    print('w', model.w)
    print('b', model.b)
    
    print('(0, 0) =>', model.forward([0, 0]))
    print('(5, 5) =>', model.forward([5, 5]))

    x = np.arange(-3,7)
    y = (-model.b - model.w[0] * x) / model.w[1]
    plt.scatter(x1.T[0], x1.T[1])
    plt.scatter(x2.T[0], x2.T[1])
    plt.plot(x, y)
    plt.show()
    