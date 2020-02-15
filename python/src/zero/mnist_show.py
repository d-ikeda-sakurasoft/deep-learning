import sys, os
sys.path.append(os.pardir)
import numpy as np
from keras.datasets import mnist
from PIL import Image
#from keras.utils import to_categorical

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.save("mnist_show.png")

(x_train, t_train), (x_test, t_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
