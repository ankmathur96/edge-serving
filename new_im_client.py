import requests
import numpy as np

im = np.random.randint(0, 256, size=(1, 299, 299, 3))
print(im.shape)

r = requests.post('http://104.196.229.77:8501/v1/models/partial_inception_v1:predict', data={'images' : im})
print(r)