import sys, os
sys.path.append(os.pardir)
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

(x_train, t_train), (x_test, t_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32')
x_train /= 255
t_train = to_categorical(t_train)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
