from pandas import DataFrame
from qwak.model.tools import run_local

from main.model import CreditRiskModel
from main.feature_set import ENTITY_KEY

if __name__ == '__main__':
    # Create a new instance of the model
    m = CreditRiskModel()

    # Define the columns
    columns = [
            ENTITY_KEY
        ]

    # Define the data
    data = [
        [
            "45b7836f-bf7c-4039-bc9e-d33982cc1fc5"
        ]
    ]

    
    # Create the DataFrame and convert it to JSON
    df = DataFrame(data, columns=columns).to_json()
    print("Predicting for: \n\n", df)
    

    # Run local inference using the model and print the prediction
    # The run_local function is part of the qwak library and allows for local testing of the model
    prediction = run_local(m, df)
    print("\nPrediction: ", prediction)