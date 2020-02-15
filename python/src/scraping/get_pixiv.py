import os, time, json, sys, re, datetime
import numpy as np
import i2v
from pixivpy3 import PixivAPI, AppPixivAPI
from PIL import Image
from dateutil.relativedelta import relativedelta

illust_re = re.compile("([0-9]+[^/]+$)")

illust2vec = i2v.make_i2v_with_chainer(
    "illust2vec_tag_ver200.caffemodel", "tag_list.json")

root_dir = "../pixiv"

#ログイン
api = PixivAPI()
aapi = AppPixivAPI()
api.login(os.environ["PIXIV_USER"], os.environ["PIXIV_PASSWORD"])

checked_user_ids = []

#取得済みのJSONはキャッシュ
def get_json(json_name, init_func):
    if os.path.isfile(json_name) == False:
        res = init_func()
        if "status" in res and res["status"] == "failure":
            sys.exit(res)
        time.sleep(1)
        with open(json_name, mode="w") as f:
            json.dump(res, f)
    with open(json_name) as f:
        return json.load(f)

#イラストタグのチェック
def contains_tags(tags):
    for tag in tags:
        if tag in ["艦これ","艦隊これくしょん"]:
            return True
    return False

def download_from_user(user_id, score=10000, ymdr="", per_page=1000):
    if user_id in checked_user_ids:
        return
    checked_user_ids.append(user_id)

    #ユーザー情報とイラスト一覧を取得
    user_dir = "%s/illusts/%d"%(root_dir,user_id)
    os.makedirs(user_dir, exist_ok=True)
    works = get_json("%s/%d.json"%(user_dir,user_id),
        lambda: api.users_works(user_id, per_page=per_page))
    checked_path = "%s/%d-checked.json"%(user_dir,user_id)
    checked = get_json(checked_path, lambda: [])

    #イラスト一覧からイラストを取得
    if "pagination" in works:
        total = works["pagination"]["total"]
        total = np.min([per_page, total])
        for j in range(0, total):
            print("\r%s %d/%d %d "%(ymdr,j+1,total,user_id), end="")
            sys.stdout.flush()

            #チェック済みか確認
            illust = works["response"][j]
            large_url = illust["image_urls"]["large"]
            small_url = illust["image_urls"]["px_480mw"]
            large_name = illust_re.search(large_url)
            small_name = illust_re.search(small_url)
            if large_name == None or small_name == None:
                continue
            large_name = large_name.group(1)
            small_name = small_name.group(1)
            if large_name in checked:
                continue

            #フォーマットや評価をチェック
            #if illust["is_manga"]:
                #continue
            if illust["type"] != "illustration":
                continue
            if illust["stats"]["score"] < score:
                continue
            if illust["age_limit"] != "all-age":
                continue
            #if illust["sanity_level"] != "white":
                #continue
            if illust["width"]*0.8 > illust["height"]:
                continue
            if illust["width"] < illust["height"]*0.6:
                continue
            #if contains_tags(illust["tags"]) == False:
                #continue

            #縮小サイズをダウンロード
            aapi.download(small_url, "%s/"%user_dir)
            start = time.time()

            #縮小サイズからタグ予測をチェック
            small_path = "%s/%s"%(user_dir,small_name)
            img = Image.open(small_path)
            tags = illust2vec.estimate_specific_tags([img], ["1girl","2girls","monochrome","comic"])[0]
            os.remove(small_path)
            if (tags["1girl"] >= 0.8 or tags["2girls"] >= 0.8) and tags["monochrome"] < 0.8:

                #待機して原寸サイズをダウンロード
                time.sleep(np.max([0, 0.5 - float(time.time() - start)]))
                aapi.download(large_url, "%s/"%user_dir)
                start = time.time()
            
            #チェック済みにする
            checked.append(large_name)
            with open(checked_path, mode="w") as f:
                json.dump(checked, f)
            
            #待機
            time.sleep(np.max([0, 0.5 - float(time.time() - start)]))
        os.system("find %s -type f | grep -e jpg -e png | wc -l"%root_dir)

def download_from_ranking():
    mode = "monthly"
    per_page = 10
    start = datetime.date(year=2019, month=5, day=1)
    end = datetime.date(year=2017, month=1, day=1)
    delta = relativedelta(days=1)

    while start >= end:

        #ランキングを取得
        ymd = str(start)
        ranking_dir = "%s/ranking"%root_dir
        os.makedirs(ranking_dir, exist_ok=True)
        ranking = get_json("%s/%s-%s-%d.json"%(ranking_dir,mode,ymd,per_page),
            lambda: api.ranking_all(mode=mode, date=ymd, per_page=per_page))
        if "response" in ranking:
            ranking = ranking["response"][0]["works"]

            #ユーザーからイラスト取得
            for i, ranker in enumerate(ranking):
                download_from_user(ranker["work"]["user"]["id"], ymdr="%s %d/%d"%(ymd,i+1,len(ranking)))
            
        #日付を遡る
        start = start - delta

def download_from_follow():
    with open("%s/follow/cute"%root_dir) as f:
        follows = re.findall("id=([0-9]+)", f.read())
        follows = [int(x) for x in follows]
        follows = list(set(follows))
    for i, user_id in enumerate(follows):
        download_from_user(user_id, ymdr="%d/%d"%(i+1,len(follows)))

#実行
#download_from_ranking()
#download_from_follow()
36924420
23098486
28992125
4094653
