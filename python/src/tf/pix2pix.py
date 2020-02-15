import sys, time, os, json
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_gan as tfgan
from PIL import Image
from google.colab import drive
drive_root = '/content/drive'
drive.mount(drive_root)
datasets_dir = "%s/My Drive/datasets"%drive_root
train_dir = "%s/My Drive/train/pix128_tpu"%drive_root
os.makedirs(train_dir, exist_ok=True)

def Unet(img_B):
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        def conv2d(x, filters, bn=True):
            x = tf.layers.conv2d(x, filters, 4, strides=2, padding='same')
            x = tf.nn.leaky_relu(x, 0.2)
            if bn:
                x = tf.layers.batch_normalization(x, momentum=0.8)
            return x
        def deconv2d(x, contracting_path, filters, drop_rate=0):
            x = tf.layers.conv2d_transpose(filters, 4, strides=2, padding='same')(x)
            x = tf.nn.relu(x)
            if drop_rate:
                x = tf.layers.dropout(x, drop_rate)
            x = tf.layers.batch_normalization(x, momentum=0.8)
            return tf.concat([x, contracting_path], axis=3)
        #エンコーダー
        c1 = conv2d(img_B, 64, False)
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
        x = tf.layers.conv2d_transpose(x, img_shape[-1], 4, padding='same')(x)
        x = tf.tanh(x)
        return x

def Discriminator(img_A, img_B):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        def d_layer(x, filters, bn=True):
            x = tf.layers.conv2d(x, filters, 4, strides=2, padding='same')
            x = tf.nn.leaky_relu(x, 0.2)
            if bn:
                x = tf.layers.batch_normalization(x, momentum=0.8)
            return x
        x = tf.concat([img_A, img_B], axis=3)
        #PatchGANのサイズまで畳み込み
        x = d_layer(x, 64, False)
        x = d_layer(x, 128)
        x = d_layer(x, 256)
        x = d_layer(x, 512)
        #0〜1ラベル出力
        x = tf.layers.conv2d(x, 1, 4, padding='same')(x)
        return x

def Pix2Pix(gen, disc, img_shape):
    img_A = Input(img_shape)
    img_B = Input(img_shape)
    fake_A = gen(img_B)
    valid = disc([fake_A, img_B])
    return Model([img_A, img_B], [valid, fake_A])

def load_datasets(path, train_num, img_shape):
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(train_num,)+img_shape)
 
def train():
    #教師データ
    train_num = 30000
    test_num = 6000
    img_size = 128
    img_shape = (img_size,img_size,3)
    train_A = load_datasets("%s/color.npy"%datasets_dir, train_num+test_num, img_shape)
    train_B = load_datasets("%s/line.npy"%datasets_dir, train_num+test_num, (img_size,img_size))
    #訓練回数
    epochs = 200
    batch_size = 100
    batch_num = train_num // batch_size
    #PatchGAN
    patch_shape = (img_size//16, img_size//16, 1)
    valid = np.ones((batch_size,) + patch_shape)
    fake = np.zeros((batch_size,) + patch_shape)
    #TPU
    tf.logging.set_verbosity(tf.logging.WARN)
    tpu_run_config = tf.compat.v1.estimator.tpu.RunConfig(
        cluster=tf.contrib.cluster_resolver.TPUClusterResolver(
            "grpc://"+os.environ["COLAB_TPU_ADDR"]),
        model_dir=train_dir,
        save_checkpoints_steps=batch_num,
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
            iterations_per_loop=batch_num,
            num_shards=8,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))
    #モデル
    if os.path.isfile(disc_path):
        gen = load_model(gen_path)
        disc = load_model(disc_path)
        print_img(1, gen, train_A, train_B, 0, train_num, "train")
        print_img(1, gen, train_A, train_B, train_num, test_num, "test")
    else:
        gen = Unet(img_shape)
        disc = Discriminator(img_shape)
        disc.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    disc.trainable = False
    pix2pix = Pix2Pix(gen, disc, img_shape)
    pix2pix.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=opt)
    #訓練
    est = tfgan.estimator.TPUGANEstimator(
        generator_fn=,
        discriminator_fn=,
        generator_loss_fn=tfang.losses.minimax_generator_loss,
        discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss,
        generator_optimizer=tf.train.AdamOptimizer(0.0002, 0.5),
        discriminator_optimizer=tf.train.AdamOptimizer(0.0002, 0.5),
        joint_train=True,  # train G and D jointly instead of sequentially.
        eval_on_tpu=True,
        train_batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.batch_size,
        predict_batch_size=_NUM_VIZ_IMAGES,
        use_tpu=FLAGS.use_tpu,
        config=config)
    #エポック
    for e in range(info["epoch"], epochs):
        start = time.time()
        #ミニバッチ
        for i in range(batch_num):
            #バッチ範囲をランダム選択
            idx = np.random.choice(train_num, batch_size, replace=False)
            imgs_A = train_A[idx].astype(np.float32) / 255
            imgs_B = convert_rgb(train_B[idx]).astype(np.float32) / 255
            #識別訓練
            fake_A = gen.predict(imgs_B)
            d_loss_real = disc.train_on_batch([imgs_A, imgs_B], valid)
            d_loss_fake = disc.train_on_batch([fake_A, imgs_B], fake)
            d_loss = np.add(d_loss_real, d_loss_fake) * 0.5
            #生成訓練
            g_loss = pix2pix.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
            #ログ
            print("\repoch:%d/%d batch:%d/%d %ds d_loss:%s g_loss:%s" %
                (e+1,epochs, (i+1),batch_num, (time.time()-start), d_loss[0], g_loss[0]), end="")
            sys.stdout.flush()
        print()
        #画像生成テスト
        print_img(e+1, gen, train_A, train_B, 0, train_num, "train")
        print_img(e+1, gen, train_A, train_B, train_num, test_num, "test")
        #重みの保存
        gen.save(gen_path)
        disc.save(disc_path)

def convert_rgb(train_B):
    return np.array([np.asarray(Image.fromarray(x).convert("RGB")) for x in train_B])

def print_img(e, gen, train_A, train_B, offset, limit, title):
    if e % 10 == 0 or e == 1:
        #データをランダム選択
        num = 10
        idx = np.random.choice(limit, num, replace=False) + offset
        imgs_A = train_A[idx]
        imgs_B = convert_rgb(train_B[idx])
        #生成してみる
        fake_A = gen.predict(imgs_B.astype(np.float32) / 255)
        fake_A = (fake_A * 255).astype(np.uint8)
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

#実行
train()
