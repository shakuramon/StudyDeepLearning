# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 21:47:00 2023

@author: owner
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping

'''
1.データの準備
'''

mnist = datasets.mnist
(x_train, t_train), (x_test, t_test) = mnist.load_data()


x_train = (x_train.reshape(-1, 784) / 255).astype(np.float32)
x_test = (x_test.reshape(-1, 784) / 255).astype(np.float32)

# 訓練データ　：　検証データ　＝　8:2
x_train, x_val, t_train, t_val = train_test_split(x_train, t_train, test_size=0.2)

'''
2.モデルの構築
'''
model = Sequential()
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

'''
3.モデルの学習
'''
#sparse_categrical_crossentropyはラベルデータをone-hotエンコーディングしない時に設定する
model.compile(optimizer='sgd', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#4.5.2追加
es = EarlyStopping(monitor='val_loss',#監視する値
                    patience = 5,#最小値を連続で上回った時に終了する回数
                    verbose = 1)#ログの出力方法

#epochごとに誤差と正解率を求めるため、validation_dataを設定
#model.fitで訓練データ・検証データの誤差・正解率を得る
hist = model.fit(x_train, t_train,
                 epochs=1000, batch_size=100,#エポック数を1000にする
                 verbose=2,
                 validation_data=(x_val, t_val),
                 callbacks=[es])#コールバックとして早期終了用インスタンスを登録

'''
4.モデルの評価
'''
#検証データの可視化
loss = hist.history['loss']
val_loss = hist.history['val_loss'] #誤差の履歴

fig = plt.figure()                  #描画領域
plt.rc('font', family='serif')      #フォント設定


plt.plot(range(len(val_loss)), val_loss,
         color='black', linewidth=1, label='val_loss') #データの描画

#4.5.2追加
plt.plot(range(len(loss)), loss,
         color='gray', linewidth=1,
         label='loss')



plt.xlabel('epochs')
plt.ylabel('loss')
# plt.savefig('output.jpg')

plt.legend()

plt.show()

#テストデータの評価
loss, acc = model.evaluate(x_test, t_test, verbose=0)
print('test_loss: {:.3f}, test_acc:{:.3f}'.format(loss, acc))




