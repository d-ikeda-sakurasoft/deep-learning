
#2乗和誤差
#引き算で単純な誤差出し、2乗で特徴を顕著にする
#合計することで全体の誤差としていると理解できる
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

#なぜ0.5を掛けるのか？
#2乗を微分すると係数が2になるのでそれを打ち消すための数学的おまじない

#one-hot表現が必要なのは損失関数で出力の要素数と合致するからと理解できる

#交差エントロピー誤差
#one-hot表現なので実質正解ラベルの出力だけを評価する
#マイナスなのだから出力がでかいほど誤差が小さい
#対数なので誤差がでかいほど顕著になると理解できる
def cross_entropy_error(y, t):
    return -np.sum(t * np.log(y))

#実用的にはlog(0)で無限大になるのを防ぐために最小数を足して実装する
def cross_entropy_error(y, t):
    return -np.sum(t * np.log(y + 1e-7))

#この2つの違いは正解ラベル以外を評価するかしないか、であると理解できる

#ミニバッチ対応版
def cross_entropy_error(y, t):

    #要素数が1でも配列として扱わせる
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]

    #batch_sizeで割るのは正規化=損失関数の平均値を出している
    #それ以外は行列計算のため同じ式で結果が変わらない
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

#one-hot表現ではない場合
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]

    #tを掛ける代わりにbatch_sizeと正解ラベルでインデックスを生成して正解に対応した出力を取り出している
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

#何故損失関数を使うのか。
#パラメータの更新は微分によって方向を決める。
#このとき認識精度を指標にすると微分がほとんど0になり調整が効かない。

#ここで思い出すべきは微分は曲線のある一点における傾きであるということ。
