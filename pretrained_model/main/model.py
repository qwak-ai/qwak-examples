import os
import pandas as pd
import qwak
from catboost import CatBoostClassifier, Pool
from qwak.model.base import QwakModelInterface
from qwak.model.schema import ModelSchema, InferenceOutput, RequestInput

RUNNING_FILE_ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


class ChurnPrediction(QwakModelInterface):

    def __init__(self):
        self.params = {
            'iterations': int(os.getenv('iterations', 500)),
            'learning_rate': float(os.getenv('learning_rate', 0.1)),
            'eval_metric': 'Accuracy',
            'random_seed': int(os.getenv('random_seed', 42)),
            'logging_level': 'Silent',
            'use_best_model': True
        }
        self.catboost = CatBoostClassifier(**self.params)
        qwak.log_param(self.params)

    def build(self):
        self.catboost.load_model(f'{RUNNING_FILE_ABSOLUTE_PATH}/model.cbm')

    def schema(self):
        return ModelSchema(
            features=[
                RequestInput(name="User_Id", type=str),
                RequestInput(name="State", type=str),
                RequestInput(name="Account_Length", type=int),
                RequestInput(name="Area_Code", type=int),
                RequestInput(name="Intl_Plan", type=int),
                RequestInput(name="VMail_Plan", type=int),
                RequestInput(name="VMail_Message", type=int),
                RequestInput(name="Day_Mins", type=float),
                RequestInput(name="Day_Calls", type=int),
                RequestInput(name="Eve_Mins", type=float),
                RequestInput(name="Eve_Calls", type=int),
                RequestInput(name="Night_Mins", type=float),
                RequestInput(name="Night_Calls", type=int),
                RequestInput(name="Intl_Mins", type=float),
                RequestInput(name="Intl_Calls", type=int),
                RequestInput(name="CustServ_Calls", type=int),
                RequestInput(name="Agitation_Level", type=int),
            ],
            inference_output=[
                InferenceOutput(name="Churn_Probability", type=float)
            ])

    @qwak.api()
    def predict(self, df):
        df = df.drop(['User_Id'], axis=1)
        return pd.DataFrame(self.catboost.predict_proba(df)[:, 1], columns=['Churn_Probability'])


if __name__ == '__main__':
    model = ChurnPrediction()
    model.build()

    prediction_df = pd.read_csv('../prediction_data.csv')
    response = model.predict(prediction_df)
    print(response)
