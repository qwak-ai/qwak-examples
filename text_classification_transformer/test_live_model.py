import argparse
import pandas as pd
from qwak_inference import RealTimeClient


def main(model_id):

    # Define the columns
    columns = ["text"]

    # Define the data
    data = [["U2 pitches for Apple New iTunes ads airing during baseball games Tuesday will feature the advertising-shy Irish rockers."]]
    
    # Create the DataFrame and convert it to JSON
    _input = pd.DataFrame(data, columns=columns).to_json(orient='records')
 
    client = RealTimeClient(model_id=model_id)
    
    print (_input)
    response = client.predict(_input)
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