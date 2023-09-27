from pandas import DataFrame
from qwak.model.tools import run_local
import json

from main import *

if __name__ == '__main__':
    # Create a new instance of the model from __init__.py
    m = load_model()

    # Define the columns
    columns = ["prompt"]

    # Define the data
    data = [["What is love?"]]
    
    # Create the DataFrame and convert it to JSON
    df = DataFrame(data, columns=columns).to_json()
    
    print("\n\nPREDICTION REQUEST:\n\n", df)

    # Run local inference using the model and print the prediction
    # The run_local function is part of the qwak library and allows for local testing of the model
    prediction = run_local(m, df)
    prediction_data = json.loads(prediction)

    # Extract the 'generated_text' value
    generated_text = prediction_data[0]['generated_text']

    print ("\n\nPREDICTION RESPONSE:\n\n" + generated_text)