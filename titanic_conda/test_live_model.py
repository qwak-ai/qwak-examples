from qwak_inference import RealTimeClient

# Please update MODEL_ID with your model id on Qwak
QWAK_MODEL_ID = 'your-model-id'

if __name__ == '__main__':
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

    client = RealTimeClient(model_id=QWAK_MODEL_ID)
    response = client.predict(feature_vector)
    print(response)
