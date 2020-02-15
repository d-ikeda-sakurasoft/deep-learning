import json
import numpy as np

with open("datasets/npy/faces.json") as f:
    faces = json.load(f)

counts = {}

for v in faces.values():
    if len(v) == 0:
        v.append((0,0,1,0))
    for _,_,k,_ in v:
        k = 2**int(np.log2(k))
        if k not in counts:
            counts[k] = 0
        counts[k] += 1

for key, w in sorted(counts.items(), key=lambda x: x[0]):
    print(key, w)
