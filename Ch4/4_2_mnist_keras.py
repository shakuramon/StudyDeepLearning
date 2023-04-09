'''
4.2.4.1 Keras (MNIST)
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import Dense,LeakyReLU # 4.3節 LeakyReLUレイヤー
from tensorflow.keras.layers import Dense, Dropout # 4.4節 DropOutレイヤー
from tensorflow.keras import backend as K # 4.3節 Swish　シグモイド関数を使うため


if __name__ == '__main__':
    np.random.seed(123)
    tf.random.set_seed(123)

    '''
    1. データの準備
    '''
    # mnistのデータ形式は以下が詳しい
    # https://weblabo.oscasierra.net/python/keras-mnist-sample.html
    
    mnist = datasets.mnist
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    
    # x_train:学習用の画像データ
    # t_train:学習用の正解データ
    # x_test:検証用の画像データ
    # t_test:検証用の正解データ

    x_train = (x_train.reshape(-1, 784) / 255).astype(np.float32)
    x_test = (x_test.reshape(-1, 784) / 255).astype(np.float32)
    t_train = np.eye(10)[t_train].astype(np.float32)
    t_test = np.eye(10)[t_test].astype(np.float32)
    
    # 生のデータ形式
    # x_train.shape(学習用の画像データ) :  (60000, 28, 28)
    # y_train_shape(学習用の正解データ) :  (60000,)
    # x_test.shape(検証用の画像データ) :  (10000, 28, 28)
    # y_test.shape(検証用の正解データ) :  (10000,)
    # 画像データは0~255の整数
    
    # reshape後
    # -1を入れると元の形から値を推測してくれる
    # x_train.shape(学習用の画像データ) :  (60000, 784)
    # y_train_shape(学習用の正解データ) :  (60000,)
    # x_test.shape(検証用の画像データ) :  (10000, 784)
    # y_test.shape(検証用の正解データ) :  (10000,)
    
    # np.eyeで10x10の単位行列生成→i∈[0,9]行目のベクトルを抜き出して1ofK表現の正解ベクトルを得る

    '''
    2. モデルの構築
    '''
    
    # 4.3節 Swish
    def swish(x, beta=1.):
        return x * K.sigmoid(beta * x)
        # return x * tf.nn.sigmoid(beta * x)  # こちらでもOK
    
    model = Sequential()
    # model.add(Dense(200, activation='sigmoid')) # 入力次元200の隠れ層を追加
    # model.add(Dense(200, activation='sigmoid'))
    # model.add(Dense(200, activation='sigmoid'))
    # model.add(Dense(20000, activation='sigmoid')) # 入力次元20000の隠れ層を追加
    # model.add(Dense(20000, activation='sigmoid'))
    # model.add(Dense(20000, activation='sigmoid'))
    # model.add(Dense(200, activation='tanh')) # 4.3節 tanh
    # model.add(Dense(200, activation='tanh'))
    # model.add(Dense(200, activation='tanh'))
    # model.add(Dense(200, activation='relu')) # 4.3節 relu
    # model.add(Dense(200, activation='relu'))
    # model.add(Dense(200, activation='relu'))
    # model.add(Dense(200)) # 4.3節 LReLU
    # model.add(LeakyReLU(0.01)) # 直前に追加したレイヤーに対して活性関数を適用する
    # model.add(Dense(200))
    # model.add(LeakyReLU(0.01))
    # model.add(Dense(200))
    # model.add(LeakyReLU(0.01))
    # model.add(Dense(200, activation=swish)) # 4.3節 Swish
    # model.add(Dense(200, activation=swish))
    # model.add(Dense(200, activation=swish))
    model.add(Dense(200, activation='relu')) # 4.4節 DropOut
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax')) # 10クラス分類の出力層

    '''
    3. モデルの学習
    '''
    model.compile(optimizer='sgd', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # モデルの設定
    # optimizerとして確率的勾配降下法(Stochastic Gradient Descent)を用いる
    # 学習率はデフォルトの0.01を使用
    # 最小化すべき誤差関数(ないし損失関数)には公差エントロピー誤差関数を用いる
    # 評価指標には正解率を用いる

    model.fit(x_train, t_train,
              epochs=30, batch_size=100,
              verbose=2)
    # モデルの学習開始

    '''
    4. モデルの評価
    '''
    loss, acc = model.evaluate(x_test, t_test, verbose=0)
    print('test_loss: {:.3f}, test_acc: {:.3f}'.format(
        loss,
        acc
    ))
