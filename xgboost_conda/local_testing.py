from qwak_inference import RealTimeClient

# Please update MODEL_ID with your model id on Qwak
QWAK_MODEL_ID = 'your-model-id'

if __name__ == '__main__':
    feature_vector = [{
        "User_Id": "0166f628-07a6-4461-9870-fc9df0df7a5b",
        "State": "CA",
        "Account_Length": 128,
        "Area_Code": 415,
        "Intl_Plan": 0,
        "VMail_Plan": 1,
        "VMail_Message": 25,
        "Day_Mins": 265,
        "Day_Calls": 110,
        "Eve_Mins": 299,
        "Eve_Calls": 10,
        "Night_Mins": 5.3,
        "Night_Calls": 10,
        "Intl_Mins": 9.6,
        "Intl_Calls": 12,
        "CustServ_Calls": 1,
        "Agitation_Level": 0,
    }]

    client = RealTimeClient(model_id=QWAK_MODEL_ID)
    response = client.predict(feature_vector)
    print(response)
