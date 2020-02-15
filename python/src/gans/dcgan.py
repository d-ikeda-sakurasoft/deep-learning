import sys, time, os, json
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from google.colab import drive

def Generator(img_size, noise_dim, class_num):
    def deconv2d(x, filters):
        x = UpSampling2D(2)(x)
        x = Conv2D(filters, 4, padding='same', activation='relu')(x)
        x = BatchNormalization(momentum=0.8)(x)
        return x
    #ラベルも入力に含める
    noise = Input((noise_dim,))
    if class_num:
        label = Input((1,), dtype=np.int32)
        label_embedding = Flatten()(Embedding(class_num, 100)(label))
        x = multiply([noise, label_embedding])
    else:
        x = noise
    #ノイズを3次元に展開
    t = img_size//16
    x = Dense(512 * t * t, activation="relu", input_dim=noise_dim)(x)
    x = Reshape((t, t, 512))(x)
    x = BatchNormalization(momentum=0.8)(x)
    #画像サイズまで逆畳み込み
    x = deconv2d(x, 512)
    x = deconv2d(x, 256)
    x = deconv2d(x, 128)
    x = deconv2d(x, 64)
    #カラーチャンネルにして出力
    x = Conv2D(3, 4, padding='same', activation='tanh')(x)
    if class_num:
        return Model([noise, label], x)
    else:
        return Model(noise, x)

def Discriminator(img_shape, class_num, cgan, acgan):
    def d_layer(x, filters, bn=True, drop=0.0):
        x = Conv2D(filters, 4, strides=2, padding='same')(x)
        x = LeakyReLU(0.2)(x)
        if drop:
            x = Dropout(drop)(x)
        if bn:
            x = BatchNormalization(momentum=0.8)(x)
        return x
    #ラベルも入力に含める
    img = Input(img_shape)
    if cgan:
        label = Input((1,), dtype=np.int32)
        label_embedding = Flatten()(Embedding(class_num, np.prod(img_shape))(label))
        flat_img = Flatten()(img)
        x = multiply([flat_img, label_embedding])
    else:
        x = img
    #PatchGANのサイズまで畳み込み
    x = d_layer(x, 64, bn=False, drop=0.25)
    x = d_layer(x, 128, drop=0.25)
    x = d_layer(x, 256, drop=0.25)
    x = d_layer(x, 512, drop=0.25)
    #0〜1ラベル出力
    x = Conv2D(1, 4, padding='same')(x)
    if cgan:
        return Model([img, label], x)
    elif acgan:
        #ラベルも出力に含める
        valid = Dense(1, activation="sigmoid")(x)
        label = Dense(class_num, activation="softmax")(x)
        return Model(img, [valid, label])
    else:
        return Model(img, x)

def create_models(gen_path, disc_path, noise_dim, class_num, cgan, acgan):
    opt = Adam(0.0002, 0.5)
    #生成モデル、識別モデル
    if os.path.isfile(disc_path):
        gen = load_model(gen_path)
        disc = load_model(disc_path)
    else:
        gen = Generator(img_size, noise_dim, class_num)
        disc = Discriminator(img_shape, class_num, cgan, acgan)
        disc.compile(loss='binary_crossentropy', optimizer=opt)
    #生成訓練モデル
    disc.trainable = False
    noise = Input((noise_dim,))
    if cgan:
        noise_label = Input((1,))
        fake_img = gen([noise, noise_label])
        valid = disc([fake_img, noise_label])
        g_trainer = Model([noise, noise_label], valid)
    elif acgan:
        noise_label = Input((1,))
        fake_img = gen([noise, noise_label])
        valid, label = disc(fake_img)
        g_trainer = Model([noise, noise_label], [valid, label])
    else:
        fake_img = gen(noise)
        valid = disc(fake_img)
        g_trainer = Model(noise, valid)
    g_trainer.compile(loss='binary_crossentropy', optimizer=opt)
 
def train(train_num, img_size, cgan=False, acgan=False):
    #ドライブをマウントしてフォルダ作成
    drive_root = '/content/drive'
    drive.mount(drive_root)
    my_drive = "%s/My Drive"%drive_root
    datasets_dir = "%s/datasets"%my_drive
    train_dir = "%s/train/face%d_%d"%(my_drive,img_size,train_num)
    if cgan: train_dir += "_cgan"
    if acgan: train_dir += "_acgan"
    imgs_dir = "%s/imgs"%train_dir
    save_dir = "%s/save"%train_dir
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    #教師データ
    img_shape = (img_size,img_size,3)
    x_train = np.memmap("%s/face%d_%d.npy"%(datasets_dir,img_size,train_num), dtype=np.uint8, mode="r", shape=(train_num,)+img_shape)
    if cgan or acgan:
        y_train = np.memmap("%s/tags%d_%d.npy"%(datasets_dir,img_size,train_num), dtype=np.uint32, mode="r", shape=(train_num))
        class_num = np.max(y_train)
    else:
        class_num = 0
    #訓練回数
    epochs = 200
    batch_size = 100
    batch_num = train_num // batch_size
    #前回までの訓練情報
    info_path = "%s/info.json"%train_dir
    info = json.load(open(info_path)) if os.path.isfile(info_path) else {"epoch":0}
    last_epoch = info["epoch"]
    #モデル
    noise_dim = 100
    gen_path = "%s/gen.h5"%train_dir
    disc_path = "%s/disc.h5"%train_dir
    gen, disc, g_trainer = create_models(gen_path, disc_path, noise_dim, class_num, cgan, acgan)
    if last_epoch:
        print_img(gen, img_size, noise_dim, class_num, imgs_dir, last_epoch)
    #PatchGAN
    patch_shape = (img_size//16, img_size//16, 1)
    real = np.ones((batch_size,) + patch_shape)
    fake = np.zeros((batch_size,) + patch_shape)
    #エポック
    for e in range(last_epoch, epochs):
        start = time.time()
        #ミニバッチ
        for i in range(batch_num):
            #バッチ範囲をランダム選択
            idx = np.random.choice(train_num, batch_size, replace=False)
            imgs = x_train[idx].astype(np.float32) / 255
            if cgan or acgan:
                labels = y_train[idx]
            #識別訓練
            #TODO noise_label
            noise = np.random.uniform(-1, 1, (batch_size, noise_dim))
            if cgan:
                fake_imgs = gen.predict([noise, labels])
                d_loss_real = disc.train_on_batch([imgs, labels], real)
                d_loss_fake = disc.train_on_batch([fake_imgs, labels], fake)
            elif acgan:
                fake_imgs = gen.predict([noise, labels])
                d_loss_real = disc.train_on_batch(imgs, [real, labels])
                d_loss_fake = disc.train_on_batch(fake_imgs, [fake, labels])
            else:
                fake_imgs = gen.predict(noise)
                d_loss_real = disc.train_on_batch(imgs, real)
                d_loss_fake = disc.train_on_batch(fake_imgs, fake)
            d_loss = np.add(d_loss_real, d_loss_fake) * 0.5
            #生成訓練
            if cgan:
                g_loss = cgan.train_on_batch([noise, labels], real)
            elif acgan:
                g_loss = acgan.train_on_batch([noise, labels], [real, labels])
            else:
                g_loss = g_trainer.train_on_batch(noise, real)
            #ログ
            print("\repoch:%d/%d batch:%d/%d %ds d_loss:%s g_loss:%s" %
                (e+1,epochs, (i+1),batch_num, (time.time()-start), d_loss, g_loss), end="")
            sys.stdout.flush()
        print()
        #画像生成テスト
        if (e+1) % 10 == 0 or e == 0:
            print_img(gen, img_size, noise_dim, class_num, imgs_dir, e+1)
        #重みの保存
        gen.save(gen_path)
        disc.save(disc_path)
        info["epoch"] += 1
        with open(info_path, "w") as f:
            json.dump(info, f)

def print_img(gen, img_size, noise_dim, class_num, imgs_dir, epoch):
    #生成してみる
    num = 30
    noise = np.random.uniform(-1, 1, (num, noise_dim))
    if class_num:
        noise_labels = np.random.randint(0, class_num, (batch_size, 1))
        imgs = gen.predict([noise, noise_labels])
    else:
        imgs = gen.predict(noise)
    imgs = (imgs * 255).clip(0).astype(np.uint8)
    #繋げる
    col = 10
    imgs = [np.concatenate(imgs[i*col:(i+1)*col], axis=1) for i in range(num//col)]
    imgs = np.concatenate(imgs, axis=0)
    #プロット
    plt.figure(figsize=(20, 6))
    plt.imshow(imgs)
    plt.axis('off')
    plt.show()
    #保存
    Image.fromarray(imgs).save("%s/%d.png"%(imgs_dir,epoch))

#実行
train(12000, 128)
#train(12000, 128, cgan=True)
#train(12000, 128, acgan=True)
