import argparse
import pandas as pd
from qwak_inference import RealTimeClient


def main(model_id):

    # Define the data
    feature_vector = [
        {
            'User_Id': 166056434,
            'State': "AZ",
            'Account_Length': 140,
            'Area_Code': 408,
            'Intl_Plan': 0,
            'VMail_Plan': 0,
            'VMail_Message': 0,
            'Day_Mins': 149.8,
            'Day_Calls': 134,
            'Eve_Mins': 164.4,
            'Eve_Calls': 98,
            'Night_Mins' : 294.7,
            'Night_Calls' : 124,
            'Intl_Mins' : 8.1,
            'Intl_Calls' : 2,
            'CustServ_Calls' : 100,
            'Agitation_Level' : 70
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