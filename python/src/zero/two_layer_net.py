import datetime
import numpy as np
import matplotlib.pyplot as plt
from external.common.functions import *
from external.common.gradient import numerical_gradient
from keras.datasets import mnist
from keras.utils import to_categorical

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads
        
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads

#net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

#x = np.random.rand(100, 784)
#t = np.random.rand(100, 10)

#grads = net.numerical_gradient(x, t)

#for i in grads:
#    print(grads[i].shape)

(x_train, t_train), (x_test, t_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32')
x_train /= 255
x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test /= 255
t_train = to_categorical(t_train)
t_test = to_categorical(t_test)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100

# 正常に学習する
learning_rate = 0.1

# 0.1より早く収束するが、過学習気味
#learning_rate = 1

# かなり歪だが一応収束する、こともある。乱数次第
#learning_rate = 10

# 発散する
#learning_rate = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(784, 50, 10)

print(datetime.datetime.now())

for i in range(iters_num):

    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for k in network.params:
        network.params[k] -= learning_rate * grad[k]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(str(i) + "/" + str(iters_num) + " train:" + str(train_acc) + " test:" + str(test_acc))

print(datetime.datetime.now())

#x = np.arange(len(train_loss_list))
#plt.plot(x, train_loss_list)

x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list)
plt.plot(x, test_acc_list)

plt.savefig("two_layer_net.png")
