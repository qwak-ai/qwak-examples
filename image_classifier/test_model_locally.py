from qwak.model.tools import run_local
import json
from PIL import Image
import numpy as np
from main import *

if __name__ == "__main__":
    # Create a new instance of the model from __init__.py

    cv = load_model()

    # Define the data

    img = Image.open('main/cat.jpeg')
    img_ndarray = np.array(img)
    img_list = img_ndarray.tolist()
    # Run local inference using the model and print the prediction
    # The run_local function is part of the qwak library and allows for local testing of the model
    
    img_json1 = json.dumps(img_list)  # This is the JSON string
    test_model = run_local(cv, img_json1)  # Pass the JSON string directly
    print(test_model)
