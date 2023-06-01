import pandas as pd
from qwak.testing.fixtures import real_time_client
from qwak_inference.realtime_client.client import InferenceOutputFormat


def test_realtime_api(real_time_client):
    feature_vector = [
        {
            "PassengerId": 762,
            "Pclass": 3,
            "Name": "Nirva, Mr. Iisakki Antino Aijo	",
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

    survived_probability: pd.DataFrame = real_time_client.predict(feature_vector, output_format=InferenceOutputFormat.PANDAS)
    assert survived_probability["Survived_Probability"].values[0] > 0
