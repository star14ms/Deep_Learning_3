import numpy as np
from parent import print
import dezero
from dezero.models import VGG16
import dezero.utils
import dezero.datasets
from PIL import Image

# 대표적인 CNN (VGG16)

# model = VGG16(pretrained=True)

# x = np.random.randn(1, 3, 224, 224).astype(np.float32)
# model.plot(x, to_file='big_step5/VGG16.png')

################################################################

url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'
img_path = dezero.utils.get_file(url)
img = Image.open(img_path)
img.show()

x = VGG16.preprocess(img)
x = x[np.newaxis]

model = VGG16(pretrained=True)
with dezero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)

model.plot(x, to_file='big_step5/vgg.pdf')
labels = dezero.datasets.ImageNet.labels()
print(labels[predict_id])
# print(labels)