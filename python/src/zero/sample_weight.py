import sys, os
sys.path.append(os.pardir)
import numpy as np
from keras.datasets import mnist
from PIL import Image
import pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    exp_a = np.exp(a - np.max(a))
    return exp_a / np.sum(exp_a)

def get_data():
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    # 前処理
    # 28x28を計算しやすいようにフラットにする
    x_test = x_test.reshape(10000, 784)
    # 画像データを0～1に正規化する
    x_test = x_test.astype('float32')
    x_test /= 255
    
    # 前処理は高速化や精度向上のために行われる
    # 0を中心に分布する、一定範囲をに収める、分布形状を均一にする(白色化)
    # 平均や標準偏差などを利用
    return x_test, t_test

def init_network():
    with open("external/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

#predict=予測
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

x, t = get_data()
network = init_network()
accuracy_cnt = 0

#rangeは0から引数までのリストを作る
#0, 1, 2, ..., len(x)
print("predict")
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

#Accuracy=正答率
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

#行列計算であるため同じネットワークで複数枚を一度に計算できる(バッチ処理)
#計算が最適化され都合効率的なのである

batch_size = 100
accuracy_cnt = 0

#rangeの引数が増えるとbatch_size飛ばしでlen(x)までのリストを作る
#0, 100, 200, ..., len(x)
print("predict batch")
for i in range(0, len(x), batch_size):
    #pythonの範囲アクセスは最大値を超えてもセーフ
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    #axisの指定で1次元目ごとにMAXの引数を取らせる
    p = np.argmax(y_batch, axis=1)
    #np配列の比較はbool配列になり、sumでtrueの個数が返る
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
