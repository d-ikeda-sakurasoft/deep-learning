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

class ConvSN2D(Conv2D):
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape, initializer=self.kernel_initializer,
            name='kernel', regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,), initializer=self.bias_initializer,
                name='bias', regularizer=self.bias_regularizer, constraint=self.bias_constraint)
        else:
            self.bias = None
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
            initializer=initializers.RandomNormal(0, 1), name='sn', trainable=False)
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
        self.built = True
    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        W_shape = self.kernel.shape.as_list()
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        sigma = K.dot(_v, W_reshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        W_bar = W_reshaped / sigma
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
        outputs = K.conv2d(inputs, W_bar, strides=self.strides,
            padding=self.padding, data_format=self.data_format, dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

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

def Discriminator(img_shape, use_ac, use_sn):
    Conv2D_or_SN = ConvSN2D if use_sn else Conv2D
    def d_layer(x, filters, bn=True, drop=0.0):
        x = Conv2D_or_SN(filters, 4, strides=2, padding='same')(x)
        x = LeakyReLU(0.2)(x)
        if drop:
            x = Dropout(drop)(x)
        if bn and not use_sn:
            x = BatchNormalization(momentum=0.8)(x)
        return x
    img_A = Input(img_shape)
    img_B = Input(img_shape)
    x = Concatenate()([img_A, img_B])
    #PatchGANのサイズまで畳み込み
    x = d_layer(x, 64, bn=False)
    x = d_layer(x, 128)
    x = d_layer(x, 256)
    x = d_layer(x, 512)
    if use_ac:
        #ラベルも出力に含める
        valid = Conv2D_or_SN(1, 4, padding='same')(x)
        label = Conv2D_or_SN(1, 4, padding='same')(x)
        return Model([img_A, img_B], [valid, label])
    else:
        #真偽出力
        x = Conv2D_or_SN(1, 4, padding='same')(x)
        return Model([img_A, img_B], x)

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

def gradient_penalty_norm(y_true, y_pred):
    gradients = K.gradients(y_true, y_pred)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    return K.sqrt(gradients_sqr_sum)

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    gradient_l2_norm = gradient_penalty_norm(y_pred, averaged_samples)
    gradient_penalty = K.square(gradient_l2_norm - 1)
    return K.mean(gradient_penalty)

def zero_centered_gradient_penalty_loss(y_true, y_pred, averaged_samples):
    gradient_l2_norm = gradient_penalty_norm(y_pred, averaged_samples)
    gradient_penalty = K.square(gradient_l2_norm)
    return K.mean(gradient_penalty)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)
    
def relativistic_loss(y_true, y_pred, real, fake, epsilon):
    fake_mean = K.mean(fake, axis=0)
    real_mean = K.mean(real, axis=0)
    mean_1 = K.mean(K.log(K.sigmoid(real - fake_mean) + epsilon), axis=-1)
    mean_2 = K.mean(K.log(1 - K.sigmoid(fake - real_mean) + epsilon), axis=-1)
    loss = mean_2 - mean_1
    return loss

def create_models(gen_path, disc_path, img_shape, use_ac, use_wgp, use_zcgp, use_sn, use_ra):
    if use_ac + use_wgp + use_zcgp + use_ra > 1:
        raise ValueError('Do not use simultaneously.')
    #生成モデル、識別モデル
    opt = Adam(0.0002, 0.5)
    use_d_trainer = use_wgp or use_ra or use_zcgp
    if os.path.isfile(disc_path):
        custom_objects = {'ConvSN2D':ConvSN2D, 'wasserstein_loss':wasserstein_loss}
        gen = load_model(gen_path, custom_objects=custom_objects)
        disc = load_model(disc_path, custom_objects=custom_objects)
    else:
        gen = Unet(img_shape)
        disc = Discriminator(img_shape, use_ac, use_sn)
        if use_ac:
            disc.compile(loss=['mean_squared_error', 'mean_squared_error'], optimizer=opt)
        elif not use_d_trainer:
            disc.compile(loss='mean_squared_error', optimizer=opt)
    #識別訓練モデル
    img_A = Input(img_shape)
    img_B = Input(img_shape)
    fake_A = gen(img_B)
    if use_d_trainer:
        gen.trainable = False
        disc.trainable = True
        fake = disc([fake_A, img_B])
        real = disc([img_A, img_B])
    if use_wgp or use_zcgp:
        ave_A = RandomWeightedAverage()([img_A, fake_A])
        ave = disc([ave_A, img_B])
        valid_loss = wasserstein_loss if use_wgp else 'mean_squared_error'
        grad_loss = gradient_penalty_loss if use_wgp else zero_centered_gradient_penalty_loss
        penalty_loss = partial(grad_loss, averaged_samples=ave)
        d_trainer = Model([img_A, img_B], [real, fake, ave])
        d_trainer.compile(loss=[valid_loss, valid_loss, penalty_loss], loss_weights=[1, 1, 10], optimizer=opt)
    elif use_ra:
        epsilon=0.000001
        rel_d_loss = partial(relativistic_loss, real=real, fake=fake, epsilon=epsilon)
        d_trainer = Model([img_A, img_B], [real, fake])
        d_trainer.compile(loss=[rel_d_loss, None], optimizer=opt)
    else:
        d_trainer = disc
    #生成訓練モデル
    gen.trainable = True
    disc.trainable = False
    if use_ac:
        valid, label = disc([fake_A, img_B])
        g_trainer = Model([img_A, img_B], [valid, label, fake_A])
        g_trainer.compile(loss=['mean_squared_error', 'mean_squared_error', 'mean_absolute_error'], loss_weights=[1, 1, 100], optimizer=opt)
    elif use_wgp or use_zcgp:
        g_trainer = Model([img_A, img_B], [fake, fake_A])
        g_trainer.compile(loss=[valid_loss, 'mean_absolute_error'], loss_weights=[1, 100], optimizer=opt)
    elif use_ra:
        rel_g_loss = partial(relativistic_loss, real=fake, fake=real, epsilon=epsilon)
        g_trainer = Model([img_A, img_B], [fake, fake_A])
        g_trainer.compile(loss=[rel_g_loss, None], optimizer=opt)
    else:
        valid = disc([fake_A, img_B])
        g_trainer = Model([img_A, img_B], [valid, fake_A])
        g_trainer.compile(loss=['mean_squared_error', 'mean_absolute_error'], loss_weights=[1, 100], optimizer=opt)
    return gen, disc, g_trainer, d_trainer

def train_on_batch(gen, g_trainer, d_trainer, train_A, train_B,
    train_num, img_size, batch_size, class_num, train_label,
    data_aug, use_ac, use_wgp, use_zcgp, use_ra):
    #PatchGAN
    patch_shape = (img_size//16, img_size//16, 1)
    ones = np.ones((batch_size,) + patch_shape)
    zeros = np.zeros((batch_size,) + patch_shape)
    real = ones
    fake = -ones if use_wgp else zeros
    dummy = zeros
    #バッチ範囲をランダム選択
    idx = np.random.choice(train_num, batch_size, replace=False)
    imgs_A = convert_rgb(train_A[idx])
    imgs_B = convert_rgb(train_B[idx])
    if data_aug and i % 2:
        imgs_A = mirror_imgs(imgs_A)
        imgs_B = mirror_imgs(imgs_B)
    imgs_A = imgs_A.astype(np.float32) / 255
    imgs_B = imgs_B.astype(np.float32) / 255
    if use_ac:
        imgs_label = train_label[idx].astype(np.float32) / class_num
        imgs_label = np.array([np.full(patch_shape, x) for x in imgs_label])
    #識別訓練
    if use_ac:
        fake_A = gen.predict(imgs_B)
        d_loss_real = d_trainer.train_on_batch([imgs_A, imgs_B], [real, imgs_label])
        d_loss_fake = d_trainer.train_on_batch([fake_A, imgs_B], [fake, imgs_label])
        d_loss = np.add(d_loss_real, d_loss_fake) * 0.5
    elif use_wgp or use_zcgp:
        d_loss = d_trainer.train_on_batch([imgs_A, imgs_B], [real, fake, dummy])
    elif use_ra:
        d_loss = d_trainer.train_on_batch([imgs_A, imgs_B], dummy)
    else:
        fake_A = gen.predict(imgs_B)
        d_loss_real = d_trainer.train_on_batch([imgs_A, imgs_B], real)
        d_loss_fake = d_trainer.train_on_batch([fake_A, imgs_B], fake)
        d_loss = np.add(d_loss_real, d_loss_fake) * 0.5
    #生成訓練
    if use_ac:
        g_loss = g_trainer.train_on_batch([imgs_A, imgs_B], [real, imgs_label, imgs_A])
    elif use_ra:
        g_loss = g_trainer.train_on_batch([imgs_A, imgs_B], dummy)
    else:
        g_loss = g_trainer.train_on_batch([imgs_A, imgs_B], [real, imgs_A])
    return d_loss, g_loss

def train(name_B, name_A, train_num, test_num, img_size, data_aug=False,
    use_ac=False, use_wgp=False, use_zcgp=False, use_sn=False, use_ra=False):
    #ドライブをマウントしてフォルダ作成
    drive_root = '/content/drive'
    drive.mount(drive_root)
    my_drive = "%s/My Drive"%drive_root
    datasets_dir = "%s/datasets"%my_drive
    train_dir = "%s/train/%s_%s%d_%d"%(my_drive,name_B,name_A,img_size,train_num)
    if data_aug: train_dir += "x2"
    if use_ac: train_dir += "_ac"
    if use_wgp: train_dir += "_wgp"
    if use_zcgp: train_dir += "_zcgp"
    if use_sn: train_dir += "_sn"
    if use_ra: train_dir += "_ra"
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
    train_label = np.memmap("%s/tags%d_%d.npy"%(datasets_dir,img_size,data_num), dtype=np.uint32, mode="r", shape=(data_num)) if use_ac else None
    class_num = np.max(train_label) if use_ac else None
    #訓練回数
    epochs = 200
    batch_size = 100
    batch_num = train_num // batch_size * (1 + data_aug)
    #前回までの訓練情報
    info_path = "%s/info.json"%train_dir
    info = json.load(open(info_path)) if os.path.isfile(info_path) else {"epoch":0}
    last_epoch = info["epoch"]
    #モデル
    gen_path = "%s/gen.h5"%train_dir
    disc_path = "%s/disc.h5"%train_dir
    gen, disc, g_trainer, d_trainer = create_models(gen_path, disc_path, img_shape, use_ac, use_wgp, use_zcgp, use_sn, use_ra)
    if last_epoch:
        print_img(gen, train_A, train_B, 0, train_num, "train", imgs_dir, last_epoch)
        print_img(gen, train_A, train_B, train_num, test_num, "test", imgs_dir, last_epoch)
    #エポック
    for e in range(last_epoch, epochs):
        start = time.time()
        #ミニバッチ
        for i in range(batch_num):
            #訓練
            d_loss, g_loss = train_on_batch(gen, g_trainer, d_trainer,
                train_A, train_B,
                train_num, img_size, batch_size,
                class_num, train_label,
                data_aug, use_ac, use_wgp, use_zcgp, use_ra)
            #ログ
            print("\repoch:%d/%d batch:%d/%d %ds d_loss:%s g_loss:%s" %
                (e+1,epochs, (i+1),batch_num, (time.time()-start), d_loss, g_loss), end="")
            sys.stdout.flush()
        print()
        #画像生成テスト
        if (e+1) % 10 == 0 or e == 0:
            print_img(gen, train_A, train_B, 0, train_num, "train", imgs_dir, e+1)
            print_img(gen, train_A, train_B, train_num, test_num, "test", imgs_dir, e+1)
            gen.save("%s/gen_%s.h5"%(save_dir,e+1))
            disc.save("%s/disc_%s.h5"%(save_dir,e+1))
        #重みの保存
        gen.save(gen_path)
        disc.save(disc_path)
        info["epoch"] += 1
        with open(info_path, "w") as f:
            json.dump(info, f)

def mirror_imgs(train_B):
    return np.array([np.asarray(ImageOps.mirror(Image.fromarray(x))) for x in train_B])

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

#フルデータセット
#train("line", "color", 40000, 10000, 128)
#train("line", "color", 40000, 10000, 128, use_wgp=True)
#train("line", "color", 40000, 10000, 128, use_zcgp=True)
train("line", "color", 40000, 10000, 128, use_ra=True)
#train("line", "color", 40000, 10000, 128, use_sn=True)
#train("line", "color", 40000, 10000, 128, use_sn=True, use_zcgp=True)
#train("line", "color", 40000, 10000, 128, use_sn=True, use_ra=True)
#train("line", "gray", 40000, 10000, 128)
#train("gray", "color", 40000, 10000, 128)

#タグ付けデータセット
#train("line", "color", 16000, 4000, 128, use_ac=True)
#train("gray", "color", 16000, 4000, 128, use_ac=True)

#背景除外データセット
#train("line", "color", 12000, 4000, 128)
