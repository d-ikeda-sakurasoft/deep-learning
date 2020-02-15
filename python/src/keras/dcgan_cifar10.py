import os, sys, math, time
import numpy as np
import matplotlib.pylab as plt
from keras.datasets import *
from keras.utils import *
from keras.models import *
from keras.layers import *
from keras.regularizers import *
from keras.optimizers import *

def generator():
    nch = 256
    reg = l1_l2(1e-7, 1e-7)
    lky = 0.2
    model = Sequential()
    model.add(Dense(nch*4*4, kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Reshape((4, 4, nch)))
    model.add(Conv2D(int(nch/2), 5, padding='same', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(LeakyReLU(lky))
    model.add(UpSampling2D(2))
    model.add(Conv2D(int(nch/2), 5, padding='same', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(LeakyReLU(lky))
    model.add(UpSampling2D(2))
    model.add(Conv2D(int(nch/4), 5, padding='same', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(LeakyReLU(lky))
    model.add(UpSampling2D(2))
    model.add(Conv2D(3, 5, padding='same', kernel_regularizer=reg))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-4, decay=1e-5))
    return model

def discriminator():
    nch = 256
    reg = l1_l2(1e-7, 1e-7)
    lky = 0.2
    model = Sequential()
    model.add(Conv2D(int(nch/4), 5, padding='same', kernel_regularizer=reg))
    model.add(SpatialDropout2D(0.5))
    model.add(MaxPooling2D(2))
    model.add(LeakyReLU(lky))
    model.add(Conv2D(int(nch/2), 5, padding='same', kernel_regularizer=reg))
    model.add(SpatialDropout2D(0.5))
    model.add(MaxPooling2D(2))
    model.add(LeakyReLU(lky))
    model.add(Conv2D(nch, 5, padding='same', kernel_regularizer=reg))
    model.add(SpatialDropout2D(0.5))
    model.add(MaxPooling2D(2))
    model.add(LeakyReLU(lky))
    model.add(Conv2D(1, 5, padding='same', kernel_regularizer=reg))
    model.add(AveragePooling2D(4))
    model.add(Flatten())
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-3, decay=1e-5))
    return model

def discriminator_on_generator(G, D):
    D.trainable = False
    model = Sequential()
    model.add(G)
    model.add(D)
    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-4, decay=1e-5))
    D.trainable = True
    return model

def train(x_original, batch_size, epochs, latent_dim):
    x_max = np.iinfo(x_original.dtype).max
    x_train = x_original.astype(np.float32) / x_max
    imshow_size = int(batch_size**0.5)
    imshow_end = imshow_size**2
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
            noise = np.random.normal(size=(batch_size, latent_dim))
            output = G.predict(noise)
            #贋作と本物のデータとラベルを1:1で用意する
            x_batch = x_train[i*batch_size : (i+1)*batch_size]
            x = np.concatenate((x_batch, output))
            y = np.concatenate((np.ones(batch_size), np.zeros(batch_size)))
            #識別モデルを訓練する
            d_loss = D.train_on_batch(x, y)
            #生成モデルを訓練する(ノイズを正解として扱った場合の識別モデルの誤差が伝播する)
            D.trainable = False
            noise = np.random.normal(size=(batch_size, latent_dim))
            g_loss = DoG.train_on_batch(noise, np.ones(batch_size))
            D.trainable = True
            #経過損失出力
            print("\repoch:%d/%d batch:%d/%d %ds d_loss:%f g_loss:%f " % (e+1, epochs, (i+1)*batch_size, train_size, (time.time() - start), d_loss, g_loss), end="")
            sys.stdout.flush()
        print()
        #エポックごとの損失推移
        d_loss_list.append(d_loss)
        g_loss_list.append(g_loss)
        #10エポックごとに経過を表示
        if (e+1) % 10 == 0:
            #損失グラフ
            t = np.arange(len(d_loss_list))
            plt.title('loss')
            plt.plot(t, d_loss_list, label='d_loss')
            plt.plot(t, g_loss_list, label='g_loss')
            plt.legend()
            plt.show()
        if (e+1) % 10 == 0 or e == 0:
            #生成された画像
            images = (output * x_max).astype(x_original.dtype)
            plt.figure(figsize=(10, 10))
            plt.title('output')
            for k, v in enumerate(images):
                if k < imshow_end:
                    plt.subplot(imshow_size, imshow_size, k+1)
                    plt.imshow(v)
                    plt.axis('off')
            plt.show()

#データ
(x_train, _), (_, _) = cifar10.load_data()

#実行
train(x_train, batch_size=100, epochs=300, latent_dim=100)
