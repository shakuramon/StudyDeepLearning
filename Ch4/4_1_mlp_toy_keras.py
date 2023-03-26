# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:12:54 2023

@author: owner
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential#model
from tensorflow.keras.layers import Dense#層
from tensorflow.keras import optimizers

if __name__ == '__main__':
    np.random.seed(123)
    tf.random.set_seed(123) #TensorFlow用のシードも指定する
    
    '''
    1.データの準備
    '''
    N = 300 #データ数300
    x, t = datasets.make_moons(N, noise=0.3)#月形データセットの作成
    t = t.reshape(N, 1)
    
    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.2)
    
    '''
    2.モデルの構築
    '''
    model = Sequential()#kerasの空modelを作成
    model.add(Dense(3, activation='sigmoid'))#層の次元３活性化関数シグモイド
    model.add(Dense(1, activation='sigmoid'))#層の次元1活性化関数シグモイド
    #入力から入れるて最後に出力層
    
    '''
    3.モデルの学習
    '''
    optimizer = optimizers.SGD(learning_rate=0.1)#確率的勾配降下法の学習率0.1
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])#modelへ勾配降下法を設定
    #binary_crossentropy(二値交差エントロピー)式3.26の交差エントロピー誤差関数と同じ
    model.fit(x_train, t_train, epochs=100, batch_size=10, verbose=1)#学習の実行（verboseはログ出力の方法）
    
    '''
    4.モデルの評価
    '''
    loss, acc = model.evaluate(x_test, t_test, verbose=0)#modelの評価までしてくれる　誤差関数の値と正解率
    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(loss, acc))