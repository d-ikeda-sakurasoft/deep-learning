import os, sys, json, re, shutil
import numpy as np
import cv2
import i2v
from PIL import Image, ImageFilter, ImageChops, ImageOps

id_re = re.compile('^[0-9]+$')
img_re = re.compile('^[0-9]+_p0.[a-z]+$')

face_cascade = cv2.CascadeClassifier(
    "../lbpcascade_animeface/lbpcascade_animeface.xml")

illust2vec = i2v.make_i2v_with_chainer(
    "illust2vec_tag_ver200.caffemodel", "tag_list.json")

datasets_dir = "../datasets/npy"
illusts_dir = "../pixiv/illusts"
tag_path = "%s/tags.json"%datasets_dir
os.makedirs(datasets_dir, exist_ok=True)

def convert_for_i2v(img):
    w, h = img.size
    m = np.min([512, h])
    return img.resize((m, int(h*m/w)), Image.LANCZOS)

def get_square(src_path):
    with Image.open(src_path) as src:
        w, h = src.size
        m = np.min([w, h])
        x = np.min([0, (m - w) // 2])
        dst = Image.new("RGB", (m, m), (255, 255, 255))
        dst.paste(src, (x, 0))
        return dst

def get_face_img(src_path, face):
    x, y, w, h = face
    src = Image.open(src_path)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    img.paste(src, (-x, -y))
    return img

def get_faces(src_path):
    src = cv2.imread(src_path)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(src)
    for face in faces:
        img = get_face_img(src_path, face)
        img = convert_for_i2v(img)
        tags = illust2vec.estimate_specific_tags([img], ["1girl", "face"])[0]
        if tags["1girl"] >= 0.8 and tags["face"] >= 0.2:
            yield face

def get_img_names(s):
    user_ids = [x for x in os.listdir(illusts_dir) if id_re.match(x)]
    for i, user_id in enumerate(user_ids):
        user_dir = "%s/%s"%(illusts_dir,user_id)
        img_names = [x for x in os.listdir(user_dir) if img_re.match(x)]
        for j, img_name in enumerate(img_names):
            print("\r%s %d/%d %d/%d %s %s " %
                (s,i+1,len(user_ids),j+1,len(img_names),user_id,img_name), end="")
            sys.stdout.flush()
            yield "%s/%s"%(user_id,img_name)
        print()

def get_json(json_name, init_func):
    if os.path.isfile(json_name):
        return json.load(open(json_name))
    return init_func()

def get_characters(tags):
    characters = [x["character"][0][0] for x in tags.values() if x["character"]]
    return sorted(list(set(characters)))

def get_general_tags(tags, key):
    generals = [x["general"] for x in tags.values() if x["general"]]
    res = [x for x in generals if x[0].endswith(key)]
    return sorted(list(set(res)))

def get_colors(tags):
    eyes = get_general_tags(tags, "eyes")
    hairs = get_general_tags(tags, "hair")
    res = eyes + hairs
    for e in eyes:
        for h in hairs:
            res.append(e+h)
    return res

def get_color(tags, img_name):
    eyes = ""
    hair = ""
    for tag in tags[img_name]["general"]:
        if tag[0].endswith("eyes"):
            eyes = tag[0]
        if tag[0].endswith("hair"):
            hair = tag[0]
    return eyes+hair

def convert_illusts(size, num, use_tags=False, simple=False, white=False):
    if simple or white or use_tags:
        tags = get_json(tag_path, lambda: {})
    if use_tags:
        tags_dst = np.memmap("%s/tags%d_%d.npy"%(datasets_dir,size,num), dtype=np.uint32, mode="w+", shape=(num))
        characters = get_characters(tags)

    color_dst = np.memmap("%s/color%d_%d.npy"%(datasets_dir,size,num), dtype=np.uint8, mode="w+", shape=(num, size, size, 3))
    gray_dst = np.memmap("%s/gray%d_%d.npy"%(datasets_dir,size,num), dtype=np.uint8, mode="w+", shape=(num, size, size))
    line_dst = np.memmap("%s/line%d_%d.npy"%(datasets_dir,size,num), dtype=np.uint8, mode="w+", shape=(num, size, size))
    k = 0

    for img_name in get_img_names("convert_illusts"):
        try:
            if use_tags:
                character = tags[img_name]["character"]
                if not character:
                    continue
            if simple or white:
                general = tags[img_name]["general"]
                if [x for x in general if x[0] in ["monochrome", "sketch"]]:
                    continue
                if simple and not [x for x in general if x[0] == "simple background"]:
                    continue
                if white and not [x for x in general if x[0] == "white background"]:
                    continue
            
            #タグ書き出し
            if use_tags:
                tags_dst[k] = characters.index(character[0][0]) + 1

            #正方形に統一
            color_img = get_square("%s/%s"%(illusts_dir,img_name))
            color_img = color_img.resize((size, size), Image.LANCZOS)
            color_dst[k] = np.asarray(color_img)

            #グレースケール
            gray_img = color_img.convert("L")
            gray_dst[k] = np.asarray(gray_img)

            #線画抽出
            expend_img = gray_img.filter(ImageFilter.MaxFilter(5))
            diff_img = ImageChops.difference(gray_img, expend_img)
            line_img = ImageOps.invert(diff_img)
            line_dst[k] = np.asarray(line_img)

            k += 1
            if k >= num:
                print()
                print("finished!")
                return
        except Exception as e:
            print("failed %s"%img_name)
            print(e)
    print("failed! %s"%k)

def convert_faces(size, num, use_tags):
    if use_tags:
        tags_dst = np.memmap("%s/face_tags%d_%d.npy"%(datasets_dir,size,num), dtype=np.uint32, mode="w+", shape=(num))
        tags = get_json(tag_path, lambda: {})
        colors = get_colors(tags)

    face_dst = np.memmap("%s/face%d_%d.npy"%(datasets_dir,size,num), dtype=np.uint8, mode="w+", shape=(num, size, size, 3))
    faces = get_json("%s/faces.json"%datasets_dir, lambda: {})
    k = 0
    
    for img_name in get_img_names("convert_faces"):
        try:
            if use_tags:
                color = get_color(tags, img_name)
                if not color:
                    continue
            if img_name not in faces:
                continue
            for face in [x for x in faces[img_name] if x[2] >= size]:

                #タグ書き出し
                if use_tags:
                    tags_dst[k] = colors.index(color) + 1

                #顔を切り抜き
                face_img = get_face_img("%s/%s"%(illusts_dir,img_name), face)
                face_img = face_img.resize((size, size), Image.LANCZOS)
                face_dst[k] = np.asarray(face_img)

                k += 1
                if k >= num:
                    print()
                    print("finished!")
                    return
        except Exception as e:
            print("failed %s"%img_name)
            print(e)
    print("failed! %s"%k)

def export_faces():
    path = "%s/faces.json"%datasets_dir
    tmp_path = "%s/faces_tmp.json"%datasets_dir
    faces = get_json(path, lambda: {})
    
    for img_name in get_img_names("export_faces"):
        try:
            #顔を検出
            if img_name not in faces:
                faces[img_name] = []
                for x, y, w, h in get_faces("%s/%s"%(illusts_dir,img_name)):
                    faces[img_name].append((int(x), int(y), int(w), int(h)))
            
            if len(faces) % 100 == 0:
                with open(tmp_path, "w") as f:
                    json.dump(faces, f)
        except Exception as e:
            print("failed %s"%img_name)
            print(e)
    with open(path, "w") as f:
        json.dump(faces, f)
    print("finished!")

def export_tags():
    tmp_path = "%s/tags_tmp.json"%datasets_dir
    tags = get_json(tag_path, lambda: {})

    for img_name in get_img_names("export_tags"):
        try:
            #タグを抽出
            if img_name not in tags:
                img = Image.open("%s/%s"%(illusts_dir,img_name))
                img = convert_for_i2v(img)
                tags[img_name] = illust2vec.estimate_plausible_tags([img], 0.1)[0]

            if len(tags) % 100 == 0:
                with open(tmp_path, "w") as f:
                    json.dump(tags, f)
        except Exception as e:
            print("failed %s"%img_name)
            print(e)
    with open(tag_path, "w") as f:
        json.dump(tags, f)
    print("finished!")

#実行
#export_tags()
#export_faces()
convert_illusts(128, 12000, white=True)
convert_illusts(512, 12000, white=True)
convert_illusts(128, 16000, simple=True)
convert_illusts(512, 16000, simple=True)
#convert_illusts(128, 20000, use_tags=True)
#convert_illusts(512, 20000, use_tags=True)
#convert_illusts(128, 50000)
#convert_illusts(512, 50000)
#convert_faces(128, 20000, use_tags=True)
#convert_faces(256, 10000, use_tags=True)
#convert_faces(128, 30000)
#convert_faces(256, 16000)

#TODO 線画をクリーンナップするためののデータセット
#TODO 画風変換するためのデータセット
#TODO 低明度を除外する
#TODO キャラよりも髪色と目の色の方が安定した学習ができそう
