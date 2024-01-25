from qwak_inference import RealTimeClient
import json
from PIL import Image
import numpy as np
import argparse


def main(model_id):
    try:
        img = Image.open('main/cat.jpeg')
        img_ndarray = np.array(img)
        img_list = img_ndarray.tolist()
        img_json = json.dumps(img_list)

        client = RealTimeClient(model_id=model_id)
        response = client.predict(img_json)
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using the following Qwak model-id.')
    parser.add_argument('model_id', type=str, help='The Qwak model ID to call for prediction.')

    args = parser.parse_args()
    main(args.model_id)

"""
USAGE:

>> python test_live_model.py <your_model_id>

"""
