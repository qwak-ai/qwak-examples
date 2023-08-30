import argparse
from qwak_inference import RealTimeClient

def main(model_id):

    input_ = {"user": "45b7836f-bf7c-4039-bc9e-d33982cc1fc5"}
    print(f"Predicting for: {input_}\n")

    client = RealTimeClient(model_id=model_id)
    response = client.predict(input_)
    print(response)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using a Qwak model.')
    parser.add_argument('model_id', type=str, help='The Qwak model ID to call for prediction.')

    args = parser.parse_args()
    main(args.model_id)
