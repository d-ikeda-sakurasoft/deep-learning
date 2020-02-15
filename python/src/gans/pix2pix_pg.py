import sys, time, os, json
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import keras.backend as K
from PIL import Image, ImageOps
from keras.models import *
from keras.layers import *
from keras.layers.merge import _Merge
from keras.optimizers import *
from google.colab import drive
from functools import partial

def Unet(img_shape, division, conv_num=7):
    def conv2d(x, conv, bn=True):
        x = conv(x)
        x = LeakyReLU(0.2)(x)
        if bn:
            x = BatchNormalization(momentum=0.8)(x)
        return x
    def deconv2d(x, conv, contracting_path):
        x = UpSampling2D(2)(x)
        x = conv(x)
        x = Activation('relu')(x)
        x = BatchNormalization(momentum=0.8)(x)
        return Concatenate()([x, contracting_path])
    #重み共有
    c = [Conv2D(min(512, 64*2**i), 4, strides=2, padding='same') for i in range(conv_num)]
    d = [Conv2D(min(512, 64*2**(conv_num-2-i)), 4, padding='same') for i in range(conv_num-1)]
    models = []
    for k in range(division):
        img_B = Input((img_shape[0]//2**k, img_shape[1]//2**k, img_shape[-1]))
        #エンコーダー
        x = [Conv2D(32*2**k, 4, padding='same')(img_B) if k > 0 else img_B]
        for i in range(conv_num-k):
            x.append(conv2d(x[-1], c[i+k], not i))
        #デコーダー
        y = x[-1]
        for i in range(conv_num-k-1):
            y = deconv2d(y, d[i], x[-2-i])
        #元サイズ出力
        y = UpSampling2D(2)(y)
        y = Conv2D(img_shape[-1], 4, padding='same', activation='tanh')(y)
        models.append(Model(img_B, y))
    return models

def Discriminator(img_shape, division, conv_num):
    def d_layer(x, conv, bn=True):
        x = conv(x)
        x = LeakyReLU(0.2)(x)
        if bn:
            x = BatchNormalization(momentum=0.8)(x)
        return x
    #重み共有
    c = [Conv2D(min(512, 64*2**i), 4, strides=2, padding='same') for i in range(conv_num)]
    models = []
    for k in range(division):
        div_shape = (img_shape[0]//2**k, img_shape[1]//2**k, img_shape[-1])
        img_A = Input(div_shape)
        img_B = Input(div_shape)
        x = Concatenate()([img_A, img_B])
        x = Conv2D(32*2**k, 4, padding='same')(x) if k > 0 else x
        #PatchGANのサイズまで畳み込み
        for i in range(conv_num-k):
            x = d_layer(x, c[i+k], not i)
        #真偽出力
        x = Conv2D(1, 4, padding='same')(x)
        models.append(Model([img_A, img_B], x))
    return models

def create_models(gen_base_path, disc_base_path, img_shape, division, disc_conv_num):
    opt = Adam(0.0002, 0.5)
    gens = Unet(img_shape, division)
    discs = Discriminator(img_shape, division, disc_conv_num)
    g_trainers = []
    for k in range(division):
        gen = gens[k]
        disc = discs[k]
        #生成訓練モデル
        disc.compile(loss='mean_squared_error', optimizer=opt)
        disc.trainable = False
        k_shape = (img_shape[0]//2**k, img_shape[1]//2**k, img_shape[-1])
        img_A = Input(k_shape)
        img_B = Input(k_shape)
        fake_A = gen(img_B)
        valid = disc([fake_A, img_B])
        g_trainer = Model([img_A, img_B], [valid, fake_A])
        g_trainer.compile(loss=['mean_squared_error', 'mean_absolute_error'], loss_weights=[1, 100], optimizer=opt)
        g_trainers.append(g_trainer)
        #重みの復元
        disc_path = "%s_%s.h5"%(disc_base_path,k)
        if os.path.isfile(disc_path):
            gen.load_weights("%s_%s.h5"%(gen_base_path,k))
            disc.load_weights(disc_path)
    return gens, discs, g_trainers, discs

def train_on_batch(gen, g_trainer, d_trainer, train_A, train_B, train_num, img_size, batch_size, patch_size):
    #PatchGAN
    patch_shape = (patch_size, patch_size, 1)
    real = np.ones((batch_size,) + patch_shape)
    fake = np.zeros((batch_size,) + patch_shape)
    #バッチ範囲をランダム選択
    idx = np.random.choice(train_num, batch_size, replace=False)
    imgs_A = convert_rgb(resize_imgs(train_A[idx], img_size)).astype(np.float32) / 255
    imgs_B = convert_rgb(resize_imgs(train_B[idx], img_size)).astype(np.float32) / 255
    #識別訓練
    fake_A = gen.predict(imgs_B)
    d_loss_real = d_trainer.train_on_batch([imgs_A, imgs_B], real)
    d_loss_fake = d_trainer.train_on_batch([fake_A, imgs_B], fake)
    d_loss = np.add(d_loss_real, d_loss_fake) * 0.5
    #生成訓練
    g_loss = g_trainer.train_on_batch([imgs_A, imgs_B], [real, imgs_A])
    return d_loss, g_loss

def train(name_B, name_A, train_num, test_num, img_size):
    #ドライブをマウントしてフォルダ作成
    drive_root = '/content/drive'
    drive.mount(drive_root)
    my_drive = "%s/My Drive"%drive_root
    datasets_dir = "%s/datasets"%my_drive
    train_dir = "%s/train/%s_%s%d_%d_pg"%(my_drive,name_B,name_A,img_size,train_num)
    imgs_dir = "%s/imgs"%train_dir
    save_dir = "%s/save"%train_dir
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    #教師データ
    img_shape = (img_size,img_size,3)
    shape_A = img_shape if name_A == "color" else (img_size,img_size)
    shape_B = img_shape if name_B == "color" else (img_size,img_size)
    data_num = train_num + test_num
    train_A = np.memmap("%s/%s%d_%d.npy"%(datasets_dir,name_A,img_size,data_num), dtype=np.uint8, mode="r", shape=(data_num,)+shape_A)
    train_B = np.memmap("%s/%s%d_%d.npy"%(datasets_dir,name_B,img_size,data_num), dtype=np.uint8, mode="r", shape=(data_num,)+shape_B)
    #訓練回数
    batch_size = 100
    disc_conv_num = 4
    batch_num = train_num // batch_size
    division_epochs = [20, 20, 20, 20]
    epochs = sum(division_epochs)
    division = len(division_epochs)
    patch_size = img_size // 2**disc_conv_num
    #前回までの訓練情報
    info_path = "%s/info.json"%train_dir
    info = json.load(open(info_path)) if os.path.isfile(info_path) else {"epoch":0}
    last_epoch = info["epoch"]
    #モデル
    gen_base_path = "%s/gen"%train_dir
    disc_base_path = "%s/disc"%train_dir
    gens, discs, g_trainers, d_trainers = create_models(gen_base_path, disc_base_path, img_shape, division, disc_conv_num)
    #エポック
    for kk, v in enumerate(division_epochs):
        k = division - kk - 1
        gen = gens[k]
        disc = discs[k]
        d_trainer = d_trainers[k]
        g_trainer = g_trainers[k]
        div_size = img_size // 2**k
        for ee in range(v):
            e = sum(division_epochs[:kk]) + ee
            if e < last_epoch:
                continue
            start = time.time()
            #ミニバッチ
            for i in range(batch_num):
                #訓練
                d_loss, g_loss = train_on_batch(gen, g_trainer, d_trainer, train_A, train_B, train_num, div_size, batch_size, patch_size)
                #ログ
                print("\repoch:%d/%d batch:%d/%d %ds d_loss:%s g_loss:%s" %
                    (e+1,epochs, (i+1),batch_num, (time.time()-start), d_loss, g_loss), end="")
                sys.stdout.flush()
            print()
            #画像生成テスト
            if (e+1) % 10 == 0 or e == 0:
                print_img(gen, train_A, train_B, 0, train_num, div_size, "train", imgs_dir, e+1)
                print_img(gen, train_A, train_B, train_num, test_num, div_size, "test", imgs_dir, e+1)
                gen.save_weights("%s/gen_%s_%s.h5"%(save_dir,k,e+1))
                disc.save_weights("%s/disc_%s_%s.h5"%(save_dir,k,e+1))
            #重みの保存
            gen.save_weights("%s_%s.h5"%(gen_base_path,k))
            disc.save_weights("%s_%s.h5"%(disc_base_path,k))
            info["epoch"] += 1
            with open(info_path, "w") as f:
                json.dump(info, f)

def mirror_imgs(train_B):
    return np.array([np.asarray(ImageOps.mirror(Image.fromarray(x))) for x in train_B])

def convert_rgb(train_B):
    if len(train_B.shape) == 3:
        return np.array([np.asarray(Image.fromarray(x).convert("RGB")) for x in train_B])
    return train_B

def resize_imgs(train_B, img_size):
    return np.array([np.asarray(Image.fromarray(x).resize((img_size, img_size))) for x in train_B])

def print_img(gen, train_A, train_B, offset, limit, img_size, title, imgs_dir, epoch):
    #データをランダム選択
    num = 10
    idx = np.random.choice(limit, num, replace=False) + offset
    imgs_A = convert_rgb(resize_imgs(train_A[idx], img_size))
    imgs_B = convert_rgb(resize_imgs(train_B[idx], img_size))
    #生成してみる
    fake_A = gen.predict(imgs_B.astype(np.float32) / 255)
    fake_A = (fake_A * 255).clip(0).astype(np.uint8)
    #繋げる
    imgs_A = np.concatenate(imgs_A, axis=1)
    imgs_B = np.concatenate(imgs_B, axis=1)
    fake_A = np.concatenate(fake_A, axis=1)
    imgs = np.concatenate((imgs_B,imgs_A,fake_A), axis=0)
    #プロット
    plt.figure(figsize=(20, 6))
    plt.title(title)
    plt.imshow(imgs)
    plt.axis('off')
    plt.show()
    #保存
    Image.fromarray(imgs).save("%s/%s_%d.png"%(imgs_dir,title,epoch))

#フルデータセット
train("line", "color", 40000, 10000, 128)
