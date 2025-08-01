from pandas import DataFrame
#from qwak.model.tools import run_local
from frogml import run_local
import json
import os

from main import load_model

if __name__ == '__main__':
    # Create a new instance of the model from __init__.py
    m = load_model()

    # Define the columns
    columns = ["text"]

    #os.environ['TRAIN'] = True

    # Define the data
    data = [["U2 pitches for Apple New iTunes ads airing during baseball games Tuesday will feature the advertising-shy Irish rockers."]]
    
    # Create the DataFrame and convert it to JSON
    df = DataFrame(data, columns=columns).to_json()

    print (df)
    
    print("\n\nPREDICTION REQUEST:\n\n", df)

    # Run local inference using the model and print the prediction
    # The run_local function is part of the qwak library and allows for local testing of the model
    prediction = run_local(m, df)
    prediction_data = json.loads(prediction)

    # Extract the 'generated_text' value
    generated_text = prediction_data

    print (f"\n\nPREDICTION RESPONSE:\n\n{generated_text}")