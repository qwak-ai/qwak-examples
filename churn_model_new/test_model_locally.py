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
            'User_Id': 166056434,
            'State': "AZ",
            'Account_Length': 140,
            'Area_Code': 408,
            'Intl_Plan': 0,
            'VMail_Plan': 0,
            'VMail_Message': 0,
            'Day_Mins': 149.8,
            'Day_Calls': 134,
            'Eve_Mins': 164.4,
            'Eve_Calls': 98,
            'Night_Mins' : 294.7,
            'Night_Calls' : 124,
            'Intl_Mins' : 8.1,
            'Intl_Calls' : 2,
            'CustServ_Calls' : 100,
            'Agitation_Level' : 70
        }]
    
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