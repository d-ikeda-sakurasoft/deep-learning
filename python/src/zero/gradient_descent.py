import numpy as np

#勾配を求める関数
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for i in range(x.size):
        v = x[i]
        
        x[i] = v + h
        f1 = f(x)
        
        x[i] = v - h
        f2 = f(x)
        
        grad[i] = (f1 - f2) / (2 * h)
        x[i] = v
    
    return grad

def function_2(x):
    return np.sum(x**2)

print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))

#勾配降下法
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for i in range(step_num):
        x -= lr * numerical_gradient(f, x)
    
    return x

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x, lr=0.1, step_num=100))
print(gradient_descent(function_2, init_x, lr=10, step_num=100))
print(gradient_descent(function_2, init_x, lr=1e-10, step_num=100))

#重要なのは微分の特性を思い出すこと
#ほんのちょっとだけxを動かしたらyがどれだけ変化するか
#傾き
#微分の数学的典型的解(x^2 > 2x, x > 1, e^x > e^x)
