import numpy as np
import matplotlib.pylab as plt

def softmax(a):
    exp_a = np.exp(a)
    return exp_a / np.sum(exp_a)

a = np.array([0.3, 2.9, 4.0])
print(softmax(a))

a = np.array([1010, 1000, 990])
print(softmax(a))

#実際には値がでかいとエラーとなる
#したがって計算結果が変わらない特質から最大の要素で分子分母全体をマイナスする
def softmax(a):
    exp_a = np.exp(a - np.max(a))
    return exp_a / np.sum(exp_a)

a = np.array([1010, 1000, 990])
y = softmax(a)
print(y)
print(np.sum(y))

#特徴は合計が1になるので確率として扱える点
#出力層に回帰問題では恒等関数、分類問題ではソフトマックスを用いるのが通例
