from pandas import DataFrame
from qwak.model.tools import run_local
from main import *

if __name__ == '__main__':
    # Create a new instance of the model
    m = load_model()

    feature_vector = [
        {
            "UserId": "male",
            "Age": 3,
            "Sex": "male",
            "Job": 2,
            "Housing": "male",
            "Saving accounts": "male",
            "Checking account": "male",
            "Credit amount": 54.2,
            "Duration": 4,
            "Purpose": "male",
            "Age_cat": "male",

        }]
    
    # Create the DataFrame and convert it to JSON
    df = DataFrame(feature_vector).to_json()
    print("Predicting for: \n\n", df)
    

    # Run local inference using the model and print the prediction
    # The run_local function is part of the qwak library and allows for local testing of the model
    prediction = run_local(m, df)
    print("\nPrediction: ", prediction)