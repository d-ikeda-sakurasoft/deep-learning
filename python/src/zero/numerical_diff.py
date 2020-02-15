import numpy as np
import matplotlib.pylab as plt

#数値微分(悪い例)
def numerical_diff(f, x):
    h = 10e-50
    return (f(x+h) - f(x)) / h

#悪い例ではhが小さすぎて誤差が生じる
#またf(x-h)との差を取ることでxを中心とした傾きになる
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

#例
def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.savefig("differential_1.png")
