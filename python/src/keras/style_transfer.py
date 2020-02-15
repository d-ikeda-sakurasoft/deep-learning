import numpy as np
import matplotlib.pylab as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
from keras import backend as K

class TransDef():
    def __init__(self, content_path, nrows=400):
        self.width, self.height = load_img(content_path).size
        self.nrows = nrows
        self.ncols = int(self.width * self.nrows / self.height)
    
    def preprocess(self, path):
        img = load_img(path, target_size=(self.nrows, self.ncols))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return img
    
    def deprocess(self, x):
        img = x.copy()
        img = img.reshape(self.nrows, self.ncols, 3)
        img[:, :, 0] += 103.939
        img[:, :, 1] += 116.779
        img[:, :, 2] += 123.68
        img = img[:, :, ::-1]
        img = np.clip(img, 0, 255).astype('uint8')
        return img

#元の画像との誤差
def content_loss(content, combination):
    return K.sum(K.square(combination - content))

#スタイルとの誤差
def style_loss(tdef, style, combination):
    def gram_matrix(x):
        assert K.ndim(x) == 3
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0 , 1)))
        gram = K.dot(features, K.transpose(features))
        return gram
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = tdef.nrows * tdef.ncols
    return K.sum(K.square(S - C)) / (4.0 * (channels**2) * (size**2))

#ピクセルの距離の誤差
def total_variation_loss(tdef, x):
    assert K.ndim(x) == 4
    a = K.square(x[:, :tdef.nrows - 1, :tdef.ncols - 1, :] - x[:, 1:, :tdef.ncols - 1, :])
    b = K.square(x[:, :tdef.nrows - 1, :tdef.ncols - 1, :] - x[:, :tdef.nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

#パス
content_path = "content.jpg"
style_path = "style.jpg"

#パラメーター
iter_count = 10
content_weight = 1.0
style_weight = 0.1
total_variation_weight = 0.001
learning_rate = 0.001

#データ
tdef = TransDef(content_path)
content_img = K.variable(tdef.preprocess(content_path))
style_img = K.variable(tdef.preprocess(style_path))
combination_img = K.placeholder((1, tdef.nrows, tdef.ncols, 3))
input_tensor = K.concatenate([content_img, style_img, combination_img], axis=0)
model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

#勾配
loss = K.variable(0.0)
feature_map = outputs_dict['block5_conv2']
feature_of_content = feature_map[0, :, :, :]
feature_of_combination = feature_map[2, :, :, :]
loss += content_weight * content_loss(feature_of_content, feature_of_combination)
feature_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
for layer_name in feature_layers:
    feature_map = outputs_dict[layer_name]
    feature_of_style = feature_map[1, :, :, :]
    feature_of_combination = feature_map[2, :, :, :]
    sl = style_loss(tdef, feature_of_style, feature_of_combination)
    loss += (style_weight / len(feature_layers)) * sl
loss += total_variation_weight * total_variation_loss(tdef, combination_img)
grads = K.gradients(loss, combination_img)[0]
style_transfer = K.function([combination_img], [loss, grads])

#勾配法
img = tdef.preprocess(content_path)
for i in range(iter_count):
    loss_value, grad_value = style_transfer([img])
    img -= grad_value * learning_rate

#表示
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(load_img(content_path, target_size=(tdef.nrows, tdef.ncols)))
plt.title("original")
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(load_img(style_path, target_size=(tdef.nrows, tdef.ncols)))
plt.title("style")
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(tdef.deprocess(img))
plt.title("styled")
plt.axis('off')
plt.tight_layout()
plt.show()
