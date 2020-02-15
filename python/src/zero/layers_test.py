from layers import *
from keras.datasets import mnist
from keras.utils import to_categorical

(x_train, t_train), (x_test, t_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32')
x_train /= 255
x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test /= 255
t_train = to_categorical(t_train)
t_test = to_categorical(t_test)

network = TwoLayerNet(784, 50, 10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))


iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

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

plt.savefig("layers.png")

#数値部分のためにlossでcross entropyは必要だが、最適化された逆伝播法では不要では？
