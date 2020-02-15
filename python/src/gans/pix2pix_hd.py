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

def conv2d(x, filters, kernel=3, strides=1, norm=True, drop=0.0):
    x = Conv2D(filters, kernel, strides=strides, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    if drop:
        x = Dropout(drop)(x)
    if norm:
        x = BatchNormalization(momentum=0.8)(x)
    return x

def deconv2d(x, filters, kernel=3, norm=True, drop=0.0):
    x = UpSampling2D(2)(x)
    x = Conv2D(filters, kernel, padding='same', activation='relu')(x)
    if drop:
        x = Dropout(drop)(x)
    if norm:
        x = BatchNormalization(momentum=0.8)(x)
    return x

def create_gens(img_shape):
    def res_block(shortcut_path, filters):
        x = conv2d(shortcut_path, filters, kernel=1)
        x = conv2d(x, filters, kernel=3)
        x = conv2d(x, filters*4, kernel=1)
        return Concatenate()([x, shortcut_path])
    def front_end(x, filters):
        i = conv2d(x, filters, norm=False)
        x = conv2d(i, filters, strides=2)
        return i, x
    def back_end(x, shortcut_path, filters):
        x = res_block(x, filters)
        x = res_block(x, filters*2)
        x = res_block(x, filters*4)
        x = res_block(x, filters*8)
        x = deconv2d(x, filters)
        z = Concatenate()([x, shortcut_path])
        y = Conv2D(img_shape[-1], 4, padding='same', activation='tanh')(z)
        return y, z
    img3 = Input(img_shape)
    img2 = Input(shrink_shape(img_shape,2))
    img1 = Input(shrink_shape(img_shape,4))
    i3, x3 = front_end(img3, 64)
    i2, x2 = front_end(img2, 32)
    i1, x1 = front_end(img1, 16)
    y1, z1 = back_end(x1, i1, 16)
    y2, z2 = back_end(Concatenate()([x2, z1]), i2, 32)
    y3, z3 = back_end(Concatenate()([x3, z2]), i3, 64)
    g1 = Model(img1, y1)
    g2 = Model([img1,img2], y2)
    g3 = Model([img1,img2,img3], y3)
    return g1, g2, g3

def create_discs(img_shape):
    def disc(img_shape, filters):
        img_A = Input(img_shape)
        img_B = Input(img_shape)
        x = Concatenate()([img_A, img_B])
        x = conv2d(x, filters, strides=2, norm=False)
        x = conv2d(x, filters*2, strides=2)
        x = conv2d(x, filters*4, strides=2)
        x = conv2d(x, filters*8, strides=2)
        x = Conv2D(1, 4, padding='same')(x)
        return Model([img_A, img_B], x)
    d1 = disc(shrink_shape(img_shape,4), 16)
    d2 = disc(shrink_shape(img_shape,2), 32)
    d3 = disc(img_shape, 64)
    return d1, d2, d3

def create_models(train_dir, img_shape, opt):
    gens = create_gens(img_shape)
    discs = create_discs(img_shape)
    g_trainers = []
    imgs = []
    for level, (gen, disc) in enumerate(zip(gens, discs), 1):
        disc.compile(loss='mean_squared_error', optimizer=opt)
        disc.trainable = False
        #生成訓練モデル
        imgs.append(Input(shrink_shape(img_shape,2**(len(gens)-level))))
        fake_A = gen(imgs)
        valid = disc([fake_A, imgs[-1]])
        g_trainer = Model(imgs, [valid, fake_A])
        g_trainers.append(g_trainer)
        #重みの復元
        disc_path = get_disc_path(train_dir, level)
        if os.path.isfile(disc_path):
            gen.load_weights(get_gen_path(train_dir, level))
            disc.load_weights(disc_path)
    return gens, discs, g_trainers

def train_on_batch(gen, g_trainer, d_trainer, train_A, train_B, train_num, img_sizes, batch_size):
    #PatchGAN
    patch_size = img_sizes[-1] // 16
    patch_shape = (patch_size, patch_size, 1)
    real = np.ones((batch_size,) + patch_shape)
    fake = np.zeros((batch_size,) + patch_shape)
    #バッチ範囲をランダム選択
    idx = np.random.choice(train_num, batch_size, replace=False)
    imgs_A = convert_rgb(resize_imgs(train_A[idx], img_sizes[-1])).astype(np.float32) / 255
    imgs_B = [convert_rgb(resize_imgs(train_B[idx], img_size)).astype(np.float32) / 255 for img_size in img_sizes]
    #識別訓練
    fake_A = gen.predict(imgs_B)
    d_loss_real = d_trainer.train_on_batch([imgs_A, imgs_B[-1]], real)
    d_loss_fake = d_trainer.train_on_batch([fake_A, imgs_B[-1]], fake)
    d_loss = np.add(d_loss_real, d_loss_fake) * 0.5
    #生成訓練
    g_loss = g_trainer.train_on_batch(imgs_B, [real, imgs_A])
    return d_loss, g_loss

def train(name_B, name_A, train_num, test_num, img_size):
    #ドライブをマウントしてフォルダ作成
    drive_root = '/content/drive'
    drive.mount(drive_root)
    my_drive = "%s/My Drive"%drive_root
    datasets_dir = "%s/datasets"%my_drive
    train_dir = "%s/train/%s_%s%d_%d_hd"%(my_drive,name_B,name_A,img_size,train_num)
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
    batch_num = train_num // batch_size
    epochs = [[0, 40], [20, 40], [20, 40]]
    #前回までの訓練情報
    info_path = "%s/info.json"%train_dir
    info = json.load(open(info_path)) if os.path.isfile(info_path) else {"epoch":0}
    last_epoch = info["epoch"]
    #モデル
    opt = Adam(0.0002, 0.5)
    gens, discs, g_trainers = create_models(train_dir, img_shape, opt)
    #エポック
    img_sizes = []
    for level, (gen, disc, g_trainer, div_epochs) in enumerate(zip(gens, discs, g_trainers, epochs), 1):
        img_sizes.append(img_size // 2**(len(gens)-level))
        for div, local_epochs in enumerate(div_epochs):
            if not local_epochs:
                continue
            #序盤は下層を固定して訓練
            for smaller_gen in gens[:level-1]:
                smaller_gen.trainable = div > 0
            g_trainer.compile(loss=['mean_squared_error', 'mean_absolute_error'], loss_weights=[1, 100], optimizer=opt)
            for local_epoch in range(1, local_epochs+1):
                epoch = np.sum(epochs[:level-1]) + np.sum(div_epochs[:div]) + local_epoch
                if epoch <= last_epoch:
                    continue
                start = time.time()
                #ミニバッチ
                for batch in range(1, batch_num+1):
                    #訓練
                    d_loss, g_loss = train_on_batch(gen, g_trainer, disc, train_A, train_B, train_num, img_sizes, batch_size)
                    #ログ
                    print("\repoch:%d/%d batch:%d/%d %ds d_loss:%s g_loss:%s" %
                        (epoch,np.sum(epochs), batch,batch_num, (time.time()-start), d_loss, g_loss), end="")
                    sys.stdout.flush()
                print()
                #画像生成テスト
                if epoch % 10 == 0 or epoch == 1:
                    print_img(gen, train_A, train_B, 0, train_num, img_sizes, "train", imgs_dir, epoch)
                    print_img(gen, train_A, train_B, train_num, test_num, img_sizes, "test", imgs_dir, epoch)
                    save_weights(gens, disc, level, save_dir, epoch)
                #重みの保存
                save_weights(gens, disc, level, train_dir)
                info["epoch"] += 1
                with open(info_path, "w") as f:
                    json.dump(info, f)

def get_gen_path(save_dir, level, epoch=None):
    return get_weights_path("gen", save_dir, level, epoch)

def get_disc_path(save_dir, level, epoch=None):
    return get_weights_path("disc", save_dir, level, epoch)

def get_weights_path(name, save_dir, level, epoch):
    if epoch:
        return "%s/%s_%s_%s.h5"%(save_dir,name,level,epoch)
    return "%s/%s_%s.h5"%(save_dir,name,level)

def save_weights(gens, disc, level, save_dir, epoch=None):
    for i, gen in enumerate(gens[:level], 1):
        if gen.trainable:
            gen.save_weights(get_gen_path(save_dir, i, epoch))
    disc.save_weights(get_disc_path(save_dir, level, epoch))

def shrink_shape(img_shape, shrink):
    return (img_shape[0]//shrink,img_shape[1]//shrink,img_shape[-1])

def mirror_imgs(train_B):
    return np.array([np.asarray(ImageOps.mirror(Image.fromarray(x))) for x in train_B])

def convert_rgb(train_B):
    if len(train_B.shape) == 3:
        return np.array([np.asarray(Image.fromarray(x).convert("RGB")) for x in train_B])
    return train_B

def resize_imgs(train_B, img_size):
    return np.array([np.asarray(Image.fromarray(x).resize((img_size, img_size))) for x in train_B])

def print_img(gen, train_A, train_B, offset, limit, img_sizes, title, imgs_dir, epoch):
    #データをランダム選択
    num = 10
    idx = np.random.choice(limit, num, replace=False) + offset
    #生成してみる
    imgs_A = convert_rgb(resize_imgs(train_A[idx], img_sizes[-1]))
    imgs_B = [convert_rgb(resize_imgs(train_B[idx], img_size)).astype(np.float32) / 255 for img_size in img_sizes]
    fake_A = gen.predict(imgs_B)
    fake_A = (fake_A * 255).clip(0).astype(np.uint8)
    imgs_B = convert_rgb(resize_imgs(train_B[idx], img_sizes[-1]))
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
