import numpy as np
import matplotlib.pylab as plt
from keras.datasets import *
from keras.utils import *
from keras.models import *
from keras.layers import *

#データ
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train  = x_train.reshape(-1, 784)
x_test   = x_test.reshape(-1, 784)
x_train  = x_train.astype('float32')
x_test   = x_test.astype('float32')
x_train /= 255
x_test  /= 255
y_train  = to_categorical(y_train, 10)
y_test   = to_categorical(y_test, 10)

#モデル
model = Sequential()
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#訓練
epochs = 10
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
plt.savefig('mnist.png')

#テスト
score = model.evaluate(x=x_test, y=y_test)
print('test_loss:', score[0])
print('test_acc:', score[1])
