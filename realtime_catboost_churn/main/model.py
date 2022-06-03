import os

import numpy as np
import pandas as pd
import qwak
from catboost import CatBoostClassifier, Pool
from qwak.model.base import QwakModelInterface
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
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
        df = pd.read_csv(f"{RUNNING_FILE_ABSOLUTE_PATH}/data.csv")

        y = df['churn']
        X = df.drop(['churn', 'User_Id', '__index_level_0__', 'event date', 'Phone'], axis=1)

        categorical_features_indices = np.where(X.dtypes != np.float64)[0]
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.25, random_state=42)

        train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
        validate_pool = Pool(X_validation, y_validation, cat_features=categorical_features_indices)

        self.catboost.fit(train_pool, eval_set=validate_pool)

        print('Simple model validation accuracy: {:.4}'.format(
            accuracy_score(y_validation, self.catboost.predict(X_validation))))
        print('Best model validation accuracy: {:.4}'.format(
            accuracy_score(y_validation, self.catboost.predict(X_validation))))

        y_predicted = self.catboost.predict(X_validation)
        f1 = f1_score(y_validation, y_predicted)
        qwak.log_data(dataframe=X, tag="train_data")
        qwak.log_metric({"f1_score" : f1})

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
