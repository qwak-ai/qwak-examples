from frogml.sdk.model.tools import run_local
import json
import pandas as pd

from main import *

if __name__ == '__main__':
    # Create a new instance of the model from __init__.py
    m = load_model()


    df = pd.read_csv("main/small_fraud_dataset.csv", nrows=1)

    # Create the DataFrame and convert it to JSON
    json_df = df.to_json()
    
    print("\n\nPREDICTION REQUEST:\n\n", df)

    # Run local inference using the model and print the prediction
    # The run_local function is part of the frogml library and allows for local testing of the model
    prediction = run_local(m, json_df)
    prediction_data = json.loads(prediction)

    # Extract the 'generated_text' value
    prediction_df = pd.DataFrame(prediction_data)

    print (f"\n\nPREDICTION RESPONSE:\n\n{prediction_df}")