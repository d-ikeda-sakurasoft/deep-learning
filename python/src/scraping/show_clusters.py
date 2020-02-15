import numpy as np
import matplotlib as plt
from sklearn.cluster import *
from sklearn.decomposition import *
from PIL import Image

def show_clusters(name, size, num, ch=None, n_clusters=3):
    if ch:
        shape = (num,size,size,ch)
    else:
        shape = (num,size,size)
    mem = np.memmap("datasets/npy/%s%d_%d.npy"%(name,size,num), dtype=np.uint8, shape=shape)
    data = np.array([np.sum(img) for img in mem])
    data = np.expand_dims(data, axis=1)
    labels = KMeans(n_clusters=n_clusters, random_state=10).fit(data).labels_
    counts = {i:np.sum(labels == i) for i in range(n_clusters)}
    print(counts)
    for i in range(n_clusters):
        imgs = mem[labels == i]
        imgs = imgs[np.random.choice(len(imgs), 100, replace=False)]
        imgs = imgs[:100]
        imgs = [np.concatenate(imgs[i*10:(i+1)*10], axis=1) for i in range(10)]
        imgs = np.concatenate(imgs, axis=0)
        imgs = Image.fromarray(imgs)
        imgs.save("datasets/sample/%s%d_%d_cls_%d-%d.jpg"%(name,size,num,n_clusters,i))

show_clusters("color", 128, 50000, ch=3, n_clusters=2)
show_clusters("color", 128, 50000, ch=3, n_clusters=3)
show_clusters("color", 128, 50000, ch=3, n_clusters=4)
