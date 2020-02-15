import numpy as np
import matplotlib.pylab as plt

def relu(x):
    return np.maximum(0, x)

print(relu(5))
print(relu(-5))

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.plot(x, y)
plt.savefig("relu.png")
