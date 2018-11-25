import requests
import json
import numpy as np

im = np.random.randint(0, 256, size=(1, 299, 299, 3))
print(im.shape)
# previous part of the model runs here:
payload = json.dumps({'instances' : im.tolist()})
r = requests.post('http://104.196.229.77:8501/v1/models/partial_inception_v1:predict', data=payload)
print(r.text)

# curl -d '{"instances": [[[[110, 166, 0], [167, 235, 75]], [[191, 166, 119], [63, 198, 166]]]]}' -X POST http://104.196.229.77:8501/v1/models/partial_inception_v1:predict