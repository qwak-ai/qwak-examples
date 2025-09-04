import pandas as pd
from frogml.core.testing.fixtures import real_time_client
from frogml_inference.realtime_client.client import InferenceOutputFormat

def test_realtime_churn(real_time_client):
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
            'Agitation_Level' : 80
        }]

    survived_probability = real_time_client.predict(feature_vector, output_format=InferenceOutputFormat.PANDAS)
    assert survived_probability['Churn_Probability'].values[0] > 0 or \
           survived_probability['Churn_Probability'].values[0] < 1