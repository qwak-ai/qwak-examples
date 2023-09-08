import argparse
import pandas as pd
from qwak_inference import RealTimeClient

from main.feature_set import ENTITY_KEY

def main(model_id):

    input_ =pd.DataFrame([{ENTITY_KEY: "45b7836f-bf7c-4039-bc9e-d33982cc1fc5"}])
    print(f"Predicting for: {input_}\n")

    client = RealTimeClient(model_id=model_id)
    response = client.predict(input_)
    print(response)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using a Qwak model.')
    parser.add_argument('model_id', type=str, help='The Qwak model ID to call for prediction.')

    args = parser.parse_args()
    main(args.model_id)



    """
    Usage:
    
    >> python main/test_live_model.py <your_model_id>
    
    """