# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression(object):
    #ロジスティック回帰
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.w = np.random.normal(size=(input_dim,))
        self.b = 0
        
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # 順伝搬
        # xはN*M行列(Mは入力ベクトルの次元)
        # N個のデータ全てに対して一気に計算する
        return sigmoid(np.matmul(x, self.w) + self.b)

    def compute_gradients(self, x, t):
        # 偏微分の計算
        # xはN*M行列、yはN次元ベクトル
        y = self.forward(x)
        delta = y - t
        dw = np.matmul(x.T, delta)
        db = np.matmul(np.ones(x.shape[0]), delta)
    
        return dw, db
    
def sigmoid(x):
    # xはN次元ベクトル
    return 1 / (1 + np.exp(-x))

def step(x):
    return 1 * (x > 0)

def graph_sigmoid():
    x = np.linspace(-10, 10)

    #シグモイド関数
    y = sigmoid(x)
    plt.plot(x, y)
    
    #シグモイド関数の微分
    dy = (1 - sigmoid(x)) * sigmoid(x)
    #plt.plot(x, dy)
    
    plt.grid()
    plt.show()
    
def graph_step():
    x = np.linspace(-10, 10,1000)

    # ステップ関数
    y = step(x)
    plt.plot(x, y)

    plt.grid()
    plt.show()
    
def graph_norm():
    x = np.linspace(-10, 10)
    # 正規分布の累積分関数を指定
    y=stats.norm(loc=0,scale=1).cdf(x)

    plt.plot(x,y)
    plt.grid()

if __name__ == '__main__':
    np.random.seed(123)
    
    '''
    1. データの準備
    '''
    # OR
    # 二次元の入力データx4 -> N=4,M=2
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    t = np.array([0, 1, 1, 1])

    '''
    2. モデルの構築
    '''
    model = LogisticRegression(input_dim=2)

    '''
    3. モデルの学習
    '''
    def compute_loss(t, y):
        # 公差エントロピー誤差関数...式(3.26)
        return (-t * np.log(y) - (1 - t) * np.log(1 - y)).sum()

    def train_step(x, t):
        # パラメータ更新
        dw, db = model.compute_gradients(x, t)
        model.w = model.w - 0.1 * dw
        model.b = model.b - 0.1 * db
        loss = compute_loss(t, model(x))
        return loss

    epochs = 100

    for epoch in range(epochs):
        # 100回のパラメータ更新
        train_loss = train_step(x, t)  # バッチ学習

        if epoch % 10 == 0 or epoch == epochs - 1:
            print('epoch: {}, loss: {:.3f}'.format(
                epoch+1,
                train_loss
            ))

    '''
    4. モデルの評価
    '''
    for input in x:
        # model(input)はinputを与えられたときにニューロンが発火する確率
        print('{} => {:.3f}'.format(input, model(input)))
        
    # シグモイド関数のグラフ
    graph_sigmoid()
    
    # ステップ関数の微分
    graph_step()
    
    # 正規分布の累積分布関数
    #graph_norm()