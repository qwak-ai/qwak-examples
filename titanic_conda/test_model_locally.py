from pandas import DataFrame
from qwak.model.tools import run_local
import json

from main import *

if __name__ == '__main__':
    # Create a new instance of the model from __init__.py
    m = load_model()

    # Define the data
    feature_vector = [
        {
            "PassengerId": 762,
            "Pclass": 3,
            "Name": "Nirva, Mr. Iisakki Antino Aijo",
            "Sex": "female",
            "Age": 34,
            "SibSp": 4,
            "Parch": 3,
            "Ticket": "a",
            "Fare": 1.0,
            "Cabin": "A",
            "Embarked": "A",
        }
    ]
    
    # Create the DataFrame and convert it to JSON
    df = DataFrame(feature_vector).to_json()
    
    print("\n\nPREDICTION REQUEST:\n\n", df)

    # Run local inference using the model and print the prediction
    # The run_local function is part of the qwak library and allows for local testing of the model
    prediction = run_local(m, df)
    prediction_data = json.loads(prediction)

    # Extract the 'generated_text' value
    generated_text = prediction_data

    print (f"\n\nPREDICTION RESPONSE:\n\n{generated_text}")