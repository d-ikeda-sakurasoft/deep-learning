import numpy as np
import matplotlib.pylab as plt
from keras.datasets import *
from keras.utils import *
from keras.models import *
from keras.layers import *

#データ
(x_train, y_train),(x_test, y_test) = cifar100.load_data()
labels = np.max(y_train) + 1
x_train  = x_train.astype('float32')
x_test   = x_test.astype('float32')
x_train /= 255
x_test  /= 255
y_train  = to_categorical(y_train, labels)
y_test   = to_categorical(y_test, labels)

#モデル
inputs = Input(shape=x_train.shape[1:])
f = 64
ki = 'he_normal'
kr = regularizers.l2(1e-11)
x = Conv2D(filters=f, kernel_size=7, padding='same', kernel_initializer=ki, kernel_regularizer=kr)(inputs)
x = MaxPooling2D(pool_size=2)(x)
n = 5
for i in range(n):
    shortcut = x
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.3)(x)
    x = Conv2D(filters=f*(2**i), kernel_size=1, padding='same', kernel_initializer=ki, kernel_regularizer=kr)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=f*(2**i), kernel_size=3, padding='same', kernel_initializer=ki, kernel_regularizer=kr)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=f*(2**(i+2)), kernel_size=1, padding='same', kernel_initializer=ki, kernel_regularizer=kr)(x)
    x = Concatenate()([x, shortcut])
    if i != (n - 1):
        x = MaxPooling2D(pool_size=2)(x)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(rate=0.4)(x)
x = Dense(units=labels, kernel_initializer=ki, kernel_regularizer=kr)(x)
x = BatchNormalization()(x)
x = Activation('softmax')(x)
x = Dropout(rate=0.4)(x)
model = Model(inputs=inputs, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#訓練
epochs = 50
history = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=100, validation_split=0.2)

#グラフ
epochs = range(epochs)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['acc'], label='training')
plt.plot(epochs, history.history['val_acc'], label='validation')
plt.title('acc')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], label='training')
plt.plot(epochs, history.history['val_loss'], label='validation')
plt.title('loss')
plt.legend()
plt.show()

#テスト
score = model.evaluate(x=x_test, y=y_test)
print('test_loss:', score[0])
print('test_acc:', score[1])
