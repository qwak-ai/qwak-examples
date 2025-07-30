import argparse
import pandas as pd
from qwak_inference import RealTimeClient


def main(model_id):

    # Load the data
 
    input_ = pd.read_csv("main/small_fraud_dataset.csv", nrows=1000)
 
    client = RealTimeClient(model_id=model_id)
    
    response = client.predict(input_)
    print(pd.DataFrame(response))


"""
USAGE:

>> python main/test_live_model.py <your_model_id>

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using the following Qwak model-id.')
    parser.add_argument('model_id', type=str, help='The Qwak model ID to call for prediction.')

    args = parser.parse_args()
    main(args.model_id)