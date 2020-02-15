import sys, time, os, json
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from google.colab import drive

def Unet(img_shape):
    def conv2d(x, filters):
        x = Conv2D(filters, 4, strides=2, padding='same')(x)
        x = LeakyReLU(0.2)(x)
        x = InstanceNormalization()(x)
        return x
    def deconv2d(x, contracting_path, filters, drop_rate=0):
        x = UpSampling2D(2)(x)
        x = Conv2D(filters, 4, padding='same', activation='relu')(x)
        if drop_rate:
            x = Dropout(drop_rate)(x)
        x = InstanceNormalization()(x)
        return Concatenate()([x, contracting_path])
    img = Input(img_shape)
    #エンコーダー
    c1 = conv2d(img, 32)
    c2 = conv2d(c1, 64)
    c3 = conv2d(c2, 128)
    #中間層
    x = conv2d(c3, 256)
    #デコーダー
    x = deconv2d(x, c3, 128)
    x = deconv2d(x, c2, 64)
    x = deconv2d(x, c1, 32)
    #元サイズ出力
    x = UpSampling2D(2)(x)
    x = Conv2D(img_shape[-1], 4, padding='same', activation='tanh')(x)
    return Model(img, x)

def Discriminator(img_shape):
    def d_layer(x, filters, bn=True):
        x = Conv2D(filters, 4, strides=2, padding='same')(x)
        x = LeakyReLU(0.2)(x)
        if bn:
            x = InstanceNormalization()(x)
        return x
    img = Input(img_shape)
    #PatchGANのサイズまで畳み込み
    x = d_layer(img, 64, False)
    x = d_layer(x, 128)
    x = d_layer(x, 256)
    x = d_layer(x, 512)
    #0〜1ラベル出力
    x = Conv2D(1, 4, padding='same')(x)
    return Model(img, x)

def CycleGAN(gen_AB, gen_BA, disc_A, disc_B, img_shape):
    img_A = Input(img_shape)
    img_B = Input(img_shape)
    fake_B = gen_AB(img_A)
    fake_A = gen_BA(img_B)
    reconstr_A = gen_BA(fake_B)
    reconstr_B = gen_AB(fake_A)
    img_A_id = gen_BA(img_A)
    img_B_id = gen_AB(img_B)
    valid_A = disc_A(fake_A)
    valid_B = disc_B(fake_B)
    return Model([img_A, img_B],
        [valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id])

def load_datasets(path, train_num, img_shape):
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(train_num,)+img_shape)

def get_json(json_name, init_func):
    if os.path.isfile(json_name):
        with open(json_name) as f:
            return json.load(f)
    else:
        return init_func()
 
def train():
    #ドライブをマウントしてフォルダ作成
    drive_root = '/content/drive'
    drive.mount(drive_root)
    datasets_dir = "%s/My Drive/datasets"%drive_root
    train_dir = "%s/My Drive/train/cycle128"%drive_root
    imgs_dir = "%s/imgs"%train_dir
    os.makedirs(imgs_dir, exist_ok=True)
    #教師データ
    train_num = 30000
    test_num = 6000
    img_size = 128
    data_num = train_num + test_num
    img_shape = (img_size,img_size,3)
    train_A = load_datasets("%s/color%d_%d.npy"%(datasets_dir,img_size,data_num), data_num, img_shape)
    train_B = load_datasets("%s/gray%d_%d.npy"%(datasets_dir,img_size,data_num), data_num, (img_size,img_size))
    #訓練回数
    epochs = 200
    batch_size = 100
    batch_num = train_num // batch_size
    #前回までの訓練情報
    info_path = "%s/info.json"%train_dir
    info = get_json(info_path, lambda: {"epoch":0})
    last_epoch = info["epoch"]
    #PatchGAN
    patch_shape = (img_size//16, img_size//16, 1)
    real = np.ones((batch_size,) + patch_shape)
    fake = np.zeros((batch_size,) + patch_shape)
    #モデル
    lambda_cycle = 10.0
    lambda_id = 0.1 * lambda_cycle
    opt = Adam(0.0002, 0.5)
    gen_AB_path = "%s/gen_AB.h5"%train_dir
    gen_BA_path = "%s/gen_BA.h5"%train_dir
    disc_A_path = "%s/disc_A.h5"%train_dir
    disc_B_path = "%s/disc_B.h5"%train_dir
    if os.path.isfile(disc_B_path):
        gen_AB = load_model(gen_AB_path, custom_objects={'InstanceNormalization': InstanceNormalization})
        gen_BA = load_model(gen_BA_path, custom_objects={'InstanceNormalization': InstanceNormalization})
        disc_A = load_model(disc_A_path, custom_objects={'InstanceNormalization': InstanceNormalization})
        disc_B = load_model(disc_B_path, custom_objects={'InstanceNormalization': InstanceNormalization})
        print_img(last_epoch, gen_BA, train_A, train_B, 0, train_num, "train", img_size)
        print_img(last_epoch, gen_BA, train_A, train_B, train_num, test_num, "test", img_size)
    else:
        gen_AB = Unet(img_shape)
        gen_BA = Unet(img_shape)
        disc_A = Discriminator(img_shape)
        disc_B = Discriminator(img_shape)
        disc_A.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
        disc_B.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    disc_A.trainable = False
    disc_B.trainable = False
    cycle_gan = CycleGAN(gen_AB, gen_BA, disc_A, disc_B, img_shape)
    cycle_gan.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
        loss_weights=[1, 1, lambda_cycle, lambda_cycle, lambda_id, lambda_id], optimizer=opt)
    #エポック
    for e in range(last_epoch, epochs):
        start = time.time()
        #ミニバッチ
        for i in range(batch_num):
            #バッチ範囲をランダム選択
            idx = np.random.choice(train_num, batch_size, replace=False)
            imgs_A = train_A[idx].astype(np.float32) / 255
            idx = np.random.choice(train_num, batch_size, replace=False)
            imgs_B = convert_rgb(train_B[idx]).astype(np.float32) / 255
            #識別訓練
            fake_B = gen_AB.predict(imgs_A)
            fake_A = gen_BA.predict(imgs_B)
            d_loss_real = disc_A.train_on_batch(imgs_A, real)
            d_loss_fake = disc_A.train_on_batch(fake_A, fake)
            d_loss_A = np.add(d_loss_real, d_loss_fake) * 0.5
            d_loss_real = disc_B.train_on_batch(imgs_B, real)
            d_loss_fake = disc_B.train_on_batch(fake_B, fake)
            d_loss_B = np.add(d_loss_real, d_loss_fake) * 0.5
            d_loss = np.add(d_loss_A, d_loss_B) * 0.5
            #生成訓練
            g_loss = cycle_gan.train_on_batch([imgs_A, imgs_B],
                [real, real, imgs_A, imgs_B, imgs_A, imgs_B])
            #ログ
            print("\repoch:%d/%d batch:%d/%d %ds d_loss:%s g_loss:%s" %
                (e+1,epochs, (i+1),batch_num, (time.time()-start), d_loss[0], g_loss[0]), end="")
            sys.stdout.flush()
        print()
        #画像生成テスト
        if (e+1) % 10 == 0 or e == 0:
            print_img(e+1, gen_BA, train_A, train_B, 0, train_num, "train", img_size)
            print_img(e+1, gen_BA, train_A, train_B, train_num, test_num, "test", img_size)
        #重みの保存
        gen_AB.save(gen_AB_path)
        gen_BA.save(gen_BA_path)
        disc_A.save(disc_A_path)
        disc_B.save(disc_B_path)
        info["epoch"] += 1
        with open(info_path, "w") as f:
            json.dump(info, f)

def convert_rgb(train_B):
    return np.array([np.asarray(Image.fromarray(x).convert("RGB")) for x in train_B])

def print_img(epoch, gen, train_A, train_B, offset, limit, title, img_size):
    #データをランダム選択
    num = 10
    idx = np.random.choice(limit, num, replace=False) + offset
    imgs_A = train_A[idx]
    imgs_B = convert_rgb(train_B[idx])
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
    Image.fromarray(imgs).save("%s/cycle%d_%d_%s.png"%(imgs_dir,img_size,epoch,title))

#実行
train()
