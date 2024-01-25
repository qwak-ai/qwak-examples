from qwak_inference import RealTimeClient
import json
from PIL import Image
import numpy as np

img = Image.open('main/cat.jpeg')
img_ndarray = np.array(img)
img_list = img_ndarray.tolist()
img_json = json.dumps(img_list)

client = RealTimeClient(model_id="image_classifier")
print(client.predict(img_json))
