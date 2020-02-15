import sys, time, os, json
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from google.colab import drive

def Unet(img_shape):
    def conv2d(x, filters, bn=True):
        x = Conv2D(filters, 4, strides=2, padding='same')(x)
        x = LeakyReLU(0.2)(x)
        if bn:
            x = BatchNormalization(momentum=0.8)(x)
        return x
    def deconv2d(x, contracting_path, filters, drop=0.0):
        x = UpSampling2D(2)(x)
        x = Conv2D(filters, 4, padding='same', activation='relu')(x)
        if drop:
            x = Dropout(drop)(x)
        x = BatchNormalization(momentum=0.8)(x)
        return Concatenate()([x, contracting_path])
    img_B = Input(img_shape)
    #エンコーダー
    c1 = conv2d(img_B, 64, bn=False)
    c2 = conv2d(c1, 128)
    c3 = conv2d(c2, 256)
    c4 = conv2d(c3, 512)
    c5 = conv2d(c4, 512)
    c6 = conv2d(c5, 512)
    #中間層
    x = conv2d(c6, 512)
    #デコーダー
    x = deconv2d(x, c6, 512)
    x = deconv2d(x, c5, 512)
    x = deconv2d(x, c4, 512)
    x = deconv2d(x, c3, 256)
    x = deconv2d(x, c2, 128)
    x = deconv2d(x, c1, 64)
    #元サイズ出力
    x = UpSampling2D(2)(x)
    x = Conv2D(img_shape[-1], 4, padding='same', activation='tanh')(x)
    return Model(img_B, x)
 
def train(name_B, name_A, train_num, test_num, img_size):
    #ドライブをマウントしてフォルダ作成
    drive_root = '/content/drive'
    drive.mount(drive_root)
    my_drive = "%s/My Drive"%drive_root
    datasets_dir = "%s/datasets"%my_drive
    train_dir = "%s/train/%s_%s%d_%d_unet"%(my_drive,name_B,name_A,img_size,train_num)
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
    epochs = 200
    batch_size = 100
    batch_num = train_num // batch_size
    #前回までの訓練情報
    info_path = "%s/info.json"%train_dir
    info = json.load(open(info_path)) if os.path.isfile(info_path) else {"epoch":0}
    last_epoch = info["epoch"]
    #TPU
    tf.keras.backend.clear_session()
    tf.logging.set_verbosity(tf.logging.WARN)
    tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu_cluster_resolver)
    #モデル
    opt = tf.train.AdamOptimizer(0.0002, 0.5)
    gen_path = "%s/gen.h5"%train_dir
    if os.path.isfile(gen_path):
        gen = load_model(gen_path)
    else:
        gen = Unet(img_shape)
        gen.compile(loss=['mae'], loss_weights=[100], optimizer=opt)
    gen = tf.contrib.tpu.keras_to_tpu_model(gen, strategy=strategy)
    if last_epoch:
        print_img(gen, train_A, train_B, 0, train_num, "train", last_epoch)
        print_img(gen, train_A, train_B, train_num, test_num, "test", last_epoch)
    #エポック
    for e in range(last_epoch, epochs):
        start = time.time()
        #ミニバッチ
        for i in range(batch_num):
            #バッチ範囲をランダム選択
            idx = np.random.choice(train_num, batch_size, replace=False)
            imgs_A = convert_rgb(train_A[idx]).astype(np.float32) / 255
            imgs_B = convert_rgb(train_B[idx]).astype(np.float32) / 255
            #生成訓練
            g_loss = gen.train_on_batch([imgs_B], [imgs_A])
            #ログ
            print("\repoch:%d/%d batch:%d/%d %ds g_loss:%s" %
                (e+1,epochs, (i+1),batch_num, (time.time()-start), g_loss[0]), end="")
            sys.stdout.flush()
        print()
        #画像生成テスト
        if (e+1) % 10 == 0 or e == 0:
            print_img(gen, train_A, train_B, 0, train_num, "train", imgs_dir, e+1)
            print_img(gen, train_A, train_B, train_num, test_num, "test", imgs_dir, e+1)
            gen.save("%s/gen_%s.h5"%(save_dir,e+1))
        #重みの保存
        gen.save(gen_path)
        info["epoch"] += 1
        with open(info_path, "w") as f:
            json.dump(info, f)

def convert_rgb(train_B):
    if len(train_B.shape) == 3:
        return np.array([np.asarray(Image.fromarray(x).convert("RGB")) for x in train_B])
    return train_B

def print_img(gen, train_A, train_B, offset, limit, title, imgs_dir, epoch):
    #データをランダム選択
    num = 10
    idx = np.random.choice(limit, num, replace=False) + offset
    imgs_A = convert_rgb(train_A[idx])
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
    Image.fromarray(imgs).save("%s/%s_%d.png"%(imgs_dir,title,epoch))

#実行
train("line", "color", 40000, 10000, 128)
