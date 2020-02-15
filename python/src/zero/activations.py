from layers import *
from keras.datasets import mnist
from keras.utils import to_categorical

x = np.random.randn(1000, 100)

node_num = 100
hidden_layer_size = 5
activations = {}

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]
    
    #w = np.random.randn(node_num, node_num) * 0.1
    #w = np.random.randn(node_num, node_num) * 0.01
    #w = np.random.randn(node_num, node_num) / np.sqrt(node_num)
    w = np.random.randn(node_num, node_num) * np.sqrt(2 / node_num)
    
    z = np.dot(x, w)
    #a = sigmoid(z)
    a = relu(z)
    activations[i] = a

plt.figure(figsize=(20, 5))

for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.hist(a.flatten(), 30, range=(0,1))

plt.savefig("activations.png")
