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
train_dir = "%s/My Drive/train/dcgan128_tpu"%drive_root
os.makedirs(train_dir, exist_ok=True)

def generator(noise, mode):
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        x = tf.layers.dense(noise, 4 * 4 * 256)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.reshape(x, [-1, 4, 4, 256])
        x = tf.layers.conv2d_transpose(x, 128, 5, 2, padding='same')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d_transpose(x, 64, 4, 2, padding='same')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d_transpose(x, 3, 4, 2, padding='same')
        x = tf.tanh(x)
        return x

def discriminator(imgs, unused_conditioning):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        x = tf.layers.conv2d(imgs, 64, 5, 2, padding='same')
        x = tf.nn.leaky_relu(x, 0.2)
        x = tf.layers.conv2d(x, 128, 5, 2, padding='same')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.leaky_relu(x, 0.2)
        x = tf.layers.conv2d(x, 256, 5, 2, padding='same')
        x = tf.layers.batch_normalization(x)
        x = tf.nn.leaky_relu(x, 0.2)
        x = tf.reshape(x, [-1, 4 * 4 * 256])
        x = tf.layers.dense(x, 1)
        return x

def train():
    #教師データ
    train_num = 10000
    test_num = 2000
    img_size = 128
    data_num = train_num + test_num
    img_shape = (img_size,img_size,3)
    imgs = load_datasets("%s/face%d_%d.npy"%(datasets_dir,img_size,data_num), data_num, img_shape)
    #訓練回数
    epochs = 200
    batch_size = 1000
    batch_num = train_num // batch_size
    step_num = epochs * batch_num
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
    #TFGAN
    est = tfgan.estimator.TPUGANEstimator(
        generator_fn=generator,
        discriminator_fn=discriminator,
        generator_loss_fn=tfang.losses.minimax_generator_loss,
        discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss,
        generator_optimizer=tf.train.AdamOptimizer(0.0002, 0.5),
        discriminator_optimizer=tf.train.AdamOptimizer(0.0002, 0.5),
        joint_train=True,
        eval_on_tpu=True,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        predict_batch_size=80,
        use_tpu=True,
        config=tpu_run_config)
    #入力ノイズ
    def input_noise(params):
        return tf.random_normal([batch_size, 64])
    def input_fn(mode, params):
        return imgs[np.random.choice(train_num, batch_size)], tf.random_normal([batch_size, 64])
    #ステップ実行
    step = estimator._load_global_step_from_checkpoint_dir(train_dir)
    while step < step_num:
        start = time.time()
        step = min(step + batch_size, step_num)
        #訓練
        est.train(input_fn=input_fn(imgs, train_num, batch_size), max_steps=step)
        #ログ
        print("\rstep:%d/%d %ds" % (step,step_num, time.time()-start), end="")
        sys.stdout.flush()
        #画像生成テスト
        print_img(step, batch_num, est)

def load_datasets(path, train_num, img_shape):
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(train_num,)+img_shape)

def print_img(step, batch_num, est):
    if step % batch_num == 0 or step == 0:
        #生成してみる
        imgs = est.predict(input_fn=input_noise(batch_size))
        imgs = [img['generated_data'][:, :, :] for img in imgs]
        imgs = (imgs * 255).astype(np.uint8)
        #繋げる
        imgs = [np.concatenate(imgs[i*10:(i+1)*10], axis=1) for i in range(10)]
        imgs = np.concatenate(imgs, axis=0)
        #プロット
        plt.figure(figsize=(20, 6))
        plt.title(title)
        plt.imshow(imgs)
        plt.axis('off')
        plt.show()

#実行
train()
