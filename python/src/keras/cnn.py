import numpy as np
import matplotlib.pylab as plt
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

#データ
(x_train, y_train),(x_test, y_test) = cifar10.load_data()
x_train  = x_train.astype('float32')
x_test   = x_test.astype('float32')
x_train /= 255
x_test  /= 255
y_train  = to_categorical(y_train, 10)
y_test   = to_categorical(y_test, 10)

#モデル
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=10))
model.add(Dropout(rate=0.5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#訓練
epochs = 20
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
plt.savefig('cnn.png')

#テスト
score = model.evaluate(x=x_test, y=y_test)
print('test_loss:', score[0])
print('test_acc:', score[1])
