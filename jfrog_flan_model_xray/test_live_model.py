import argparse
from qwak_inference import RealTimeClient


def main(model_id):

    input_ = [{
        "prompt": "Question: What are three great types of food?"
    }]
     
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

    while True:
        main(args.model_id)