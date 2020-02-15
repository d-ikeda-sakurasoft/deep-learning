import glob
import numpy as np
from PIL import Image

def show_touch(name, size, num, ch=None):
    if ch:
        shape = (num,size,size,ch)
    else:
        shape = (num,size,size)
    imgs = np.memmap("datasets/npy/%s%d_%d.npy"%(name,size,num), dtype=np.uint8, shape=shape)
    imgs = imgs[-100:]
    imgs = imgs[:10]
    std = [np.std(img.flatten()) for img in imgs]
    num = [len(set(img.flatten())) for img in imgs]
    print(std)
    print(num)
    #背景を取り除かないと正確な結果が得られない

show_touch("gray", 128, 20000)
