from pandas import DataFrame
from qwak.model.tools import run_local

from main import *

if __name__ == '__main__':
    # Create a new instance of the model from __init__.py
    m = load_model()

    # Define the columns
    columns = [
            "text"
        ]

    # Define the data
    data = [
        ["Would you like some coffee?"],
        ["I love cats and also dogs."],
        ["I feel a bit under the weather today and I hate it."]
    ]

    
    # Create the DataFrame and convert it to JSON
    df = DataFrame(data, columns=columns).to_json()
    
    print("\n\nPREDICTION REQUEST:\n\n", df)
    

    # Run local inference using the model and print the prediction
    # The run_local function is part of the qwak library and allows for local testing of the model
    prediction = run_local(m, df)

    print ("\n\nPREDICTION RESPONSE:\n\n" + prediction)