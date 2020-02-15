import os
import numpy as np
from PIL import Image

def show_datasets(name, size, num, ch=None):
    if ch:
        shape = (num,size,size,ch)
    else:
        shape = (num,size,size)
    imgs = np.memmap("datasets/npy/%s%d_%d.npy"%(name,size,num), dtype=np.uint8, shape=shape)
    #imgs = imgs[np.random.choice(num, 100, replace=False)]
    imgs = imgs[-100:]
    #imgs = imgs[:100]
    imgs = [np.concatenate(imgs[i*10:(i+1)*10], axis=1) for i in range(10)]
    imgs = np.concatenate(imgs, axis=0)
    imgs = Image.fromarray(imgs)
    imgs.save("datasets/sample/%s%d_%d.jpg"%(name,size,num))

os.makedirs("datasets/sample", exist_ok=True)
show_datasets("color", 128, 12000, 3)
show_datasets("color", 512, 12000, 3)
show_datasets("color", 128, 16000, 3)
show_datasets("color", 512, 16000, 3)
#show_datasets("gray", 128, 20000)
#show_datasets("color", 128, 20000, 3)
#show_datasets("color", 512, 20000, 3)
#show_datasets("color", 128, 50000, 3)
#show_datasets("color", 512, 50000, 3)
#show_datasets("face", 128, 30000, 3)
#show_datasets("face", 256, 16000, 3)
