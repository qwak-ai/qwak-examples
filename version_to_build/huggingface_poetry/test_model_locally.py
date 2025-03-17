from frogml_core.model.tools import run_local
from pandas import DataFrame
from main import *

if __name__ == '__main__':
    # Create a new instance of the model
    m = load_model()

    feature_vector = [
        {
            "text": "An example sentence"
        }]
    # Create the DataFrame and convert it to JSON
    df = DataFrame(feature_vector).to_json()
    print("Predicting for: \n\n", df)

    # Run local inference using the model and print the prediction
    # The run_local function is part of the qwak library and allows for local testing of the model
    prediction = run_local(m, df)
    print("\nPrediction: ", prediction)