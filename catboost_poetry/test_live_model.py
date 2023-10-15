import argparse
import pandas as pd
from qwak_inference import RealTimeClient


def main(model_id):

    # Define the data
    feature_vector = [
        {
            "UserId": "male",
            "Age": 3,
            "Sex": "male",
            "Job": 2,
            "Housing": "male",
            "Saving accounts": "male",
            "Checking account": "male",
            "Credit amount": 54.2,
            "Duration": 4,
            "Purpose": "male",
            "Age_cat": "male",

        }]

    input_ = pd.DataFrame(feature_vector)    
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