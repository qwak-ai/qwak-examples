import argparse
import pandas as pd
from qwak_inference import RealTimeClient


def main(model_id):

    # Define the columns
    columns = [
            "text"
        ]

    # Define the data
    data = [
        ["I love this product. It's amazing!"],
        ["The service was terrible. I'm disappointed."],
        ["The food was okay, not great but not bad either."]
    ]

    input_ = pd.DataFrame(data, columns=columns)    
    client = RealTimeClient(model_id=model_id)
    
    response = client.predict(input_)
    print(response)


"""
USAGE:

>> python main/test_live_model.py <your_model_id>

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using the following Qwak model-id.')
    parser.add_argument('model_id', type=str, help='The Qwak model ID to call for prediction.')

    args = parser.parse_args()
    main(args.model_id)