import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread("external/dataset/lena.png")
plt.imshow(img)
plt.savefig("imshow.png")
