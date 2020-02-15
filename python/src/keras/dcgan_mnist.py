import os, sys, math, time
import numpy as np
import matplotlib.pylab as plt
from keras.datasets import *
from keras.utils import *
from keras.models import *
from keras.layers import *
from PIL import *

def generator():
    #入力を畳み込みで変換する生成モデル
    model = Sequential()
    model.add(Dense(1024, activation='tanh'))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D(2))
    model.add(Conv2D(64, 5, padding='same', activation='tanh', data_format='channels_last'))
    model.add(UpSampling2D(2))
    model.add(Conv2D(1, 5, padding='same', activation='tanh', data_format='channels_last'))
    model.compile(loss='binary_crossentropy', optimizer='SGD')
    return model

def discriminator():
    #入力が正か偽に2クラス分類する識別モデル
    model = Sequential()
    model.add(Conv2D(64, 5, padding='same', activation='tanh', data_format='channels_last'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(128, 5, activation='tanh', data_format='channels_last'))
    model.add(Flatten())
    model.add(Dense(1024, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='SGD')
    return model

def discriminator_on_generator(G, D):
    #生成モデルを訓練するための複合モデル
    model = Sequential()
    model.add(G)
    #識別モデルを通すが訓練はしない
    D.trainable = False
    model.add(D)
    model.compile(loss='binary_crossentropy', optimizer='SGD')
    D.trainable = True
    return model

def save_images(output, name):
    #生成者の出力バイナリを連結画像として保存する
    n = output.shape[0]
    w = int(math.sqrt(n))
    h = int(math.ceil(float(n) / w))
    s = output.shape[1:3]
    img = np.zeros((h*s[0], w*s[1]), dtype=output.dtype)
    for k, v in enumerate(output):
        i = int(k/w)
        j = k % w
        img[i*s[0]:(i+1)*s[0], j*s[1]:(j+1)*s[1]] = v[:, :, 0]
    img = img*127.5+127.5
    Image.fromarray(img.astype(np.uint8)).save(name)

def train(x_train, batch_size, epochs):
    #モデルの訓練を行う
    G = generator()
    D = discriminator()
    DoG = discriminator_on_generator(G, D)
    d_loss_list = []
    g_loss_list = []
    for e in range(epochs):
        start = time.time()
        train_size = x_train.shape[0]
        for i in range(int(train_size / batch_size)):
            #ノイズから画像を生成する
            output = G.predict(np.random.uniform(-1, 1, size=(batch_size, 100)))
            #贋作と本物のデータとラベルを1:1で用意する
            x = np.concatenate((x_train[i*batch_size : (i+1)*batch_size], output))
            y = np.concatenate((np.ones(batch_size), np.zeros(batch_size)))
            #識別モデルを訓練する
            d_loss = D.train_on_batch(x, y)
            #生成モデルを訓練する(ノイズを正解として扱った場合の識別モデルの誤差が伝播する)
            D.trainable = False
            g_loss = DoG.train_on_batch(np.random.uniform(-1, 1, (batch_size, 100)), np.ones(batch_size))
            D.trainable = True
            #経過損失出力
            print("\repoch:%d/%d batch:%d/%d %ds d_loss:%f g_loss:%f " % (e+1, epochs, (i+1)*batch_size, train_size, (time.time() - start), d_loss, g_loss), end="")
            sys.stdout.flush()
        print()
        if (e+1) % 10 == 0 or e == 0:
            os.makedirs("dcgan/images", exist_ok=True)
            os.makedirs("dcgan/g_weights", exist_ok=True)
            os.makedirs("dcgan/d_weights", exist_ok=True)
            #経過画像出力
            save_images(output, "dcgan/images/%03d.png" % (e+1))
            #重みを保存
            G.save_weights('dcgan/g_weights/%03d.h5' % (e+1), True)
            D.save_weights('dcgan/d_weights/%03d.h5' % (e+1), True)
        #エポックごとの損失推移
        d_loss_list.append(d_loss)
        g_loss_list.append(g_loss)
    return d_loss_list, g_loss_list

#データ
(x_train, _), (_, _) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = x_train[:, :, :, None]

#実行
d_loss_list, g_loss_list = train(x_train, batch_size=100, epochs=100)

#グラフ
x = np.arange(len(d_loss_list))
plt.plot(x, d_loss_list, label="d_loss")
plt.plot(x, g_loss_list, label="g_loss")
plt.legend()
plt.show()

#新しい生成は訓練と同様ノイズをGに渡せばよい
#結果を複数取得してDの損失が一番低いものを選び取るやり方がある
