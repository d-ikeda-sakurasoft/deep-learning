import json
import numpy as np

with open("datasets/npy/tags.json") as f:
    tags = json.load(f)

#cGANに有効なタグの選定
#姿勢(sitting, standing,)
#服装(school uniform,)
#目(blue eyes,)
#髪(long hair,)
#キャラ名(hibiki,)

#face
#monochrome
#sketch
#signature
#night
#dark skin
#simple background
#white background

rates = {}
counts = {}

for name, info in tags.items():
    for key, values in info.items():
        for v in values:
            k = "%s:%s"%(key,v[0])
            k = k.replace(" ", "_")
            if k.startswith("general"):
                if k not in rates:
                    rates[k] = 0.0
                    counts[k] = 0
                rates[k] += v[1]
                counts[k] += 1

i = 1
for key, rate in sorted(rates.items(), key=lambda x: x[1]):
    print(i, key, rate, counts[key])
    i += 1
