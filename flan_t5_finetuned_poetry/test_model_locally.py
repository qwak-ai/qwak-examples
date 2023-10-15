from pandas import DataFrame
from qwak.model.tools import run_local
import json

from main import load_model


from main import load_model

if __name__ == '__main__':
    # Create a new instance of the model from __init__.py
    m = load_model()

    # Create an input vector and convert it to JSON
    input_vector = DataFrame(
        [{
            "prompt": "Question: Answer Why does it matter if a Central Bank has a negative rather than 0% interest rate?"
        }]
    ).to_json()
    
    print("\n\nPREDICTION REQUEST:\n\n", input_vector)

    # Run local inference using the model and print the prediction
    # The run_local function is part of the qwak library and allows for local testing of the model
    prediction = run_local(m, input_vector)

    print (f"\n\nPREDICTION RESPONSE:\n\n{prediction}")

    prediction_data = json.loads(prediction)

    print (f"\n\nPREDICTION RESPONSE:\n\n{prediction_data}")