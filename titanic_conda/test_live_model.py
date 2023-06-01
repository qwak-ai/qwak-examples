from qwak_inference import RealTimeClient

# Please update MODEL_ID with your model id on Qwak
QWAK_MODEL_ID = 'titanic'

if __name__ == '__main__':
    # feature_vector = [
    #     {
    #         "PassengerId": 762,
    #         "Pclass": 3,
    #         "Name": "Nirva, Mr. Iisakki Antino Aijo	",
    #         "Sex": "female",
    #         "Age": 34,
    #         "SibSp": 4,
    #         "Parch": 3,
    #         "Ticket": "a",
    #         "Fare": 1.0,
    #         "Cabin": "A",
    #         "Embarked": "A",
    #     }
    # ]
    # from qwak_inference import RealTimeClient

    feature_vector = [
        {
            "PassengerId": 0,
            "SibSp": 0,
            "Sex": "female",
            "Name": "Nirva, Mr. Iisakki Antino Aijo",
            "Parch": 0,
            "Age": 34,
            "Pclass": 0,
            "Cabin": "A",
            "Fare": 0.0,
            "Ticket": "a",
            "Embarked": "A"
        }
    ]


    client = RealTimeClient(model_id=QWAK_MODEL_ID)
    response = client.predict(feature_vector)
    print(response)
