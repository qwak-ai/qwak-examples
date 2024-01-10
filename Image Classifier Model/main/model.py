from qwak.model.tools import run_local
import numpy as np
import pandas as pd
from PIL import Image
from qwak.model.base import QwakModel
import qwak
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from qwak.model.adapters import NumpyInputAdapter

import json


class ImageClassifier(QwakModel):
    def __init__(self):
        self._model = None

    def build(self):
        # Load the pre-trained ResNet50 model
        self._model = ResNet50(weights='imagenet')

    def preprocess_image(self, img_ndarray):
        # Check if img_ndarray is a string and convert to ndarray
        if isinstance(img_ndarray, str):
            img_list = json.loads(img_ndarray)  # Convert JSON string to Python list
            img_ndarray = np.array(img_list)    # Convert list to ndarray
        # Note: If img_ndarray is already an ndarray, this part will be skipped

        # Ensure correct format and type
        if img_ndarray.ndim == 2:
            img_ndarray = np.stack((img_ndarray,)*3, axis=-1)
        if img_ndarray.dtype != np.float32:
            img_ndarray = img_ndarray.astype('float32')

        # Resize and preprocess the image
        img_resized = Image.fromarray(np.uint8(img_ndarray)).resize((224, 224))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        return img_array

    @qwak.api(input_adapter=NumpyInputAdapter())
    def predict(self, img_ndarray: np.ndarray) -> pd.DataFrame:
        # Preprocess the image and make predictions
        preprocessed_image = self.preprocess_image(img_ndarray[0])  # assuming the adapter converts it to an ndarray
        predictions = self._model.predict(preprocessed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        top_prediction = pd.DataFrame(decoded_predictions, columns=['id', 'name', 'probability']).drop('id', axis=1)

        # Return the top prediction in a DataFrame
        return top_prediction


if __name__ == "__main__":
    cv = ImageClassifier()
    img = Image.open('cat.jpeg')
    img_ndarray = np.array(img)
    img_list = img_ndarray.tolist()
    img_json1 = json.dumps(img_list)  # This is the JSON string
    test_model = run_local(cv, img_json1)  # Pass the JSON string directly
    print(test_model)