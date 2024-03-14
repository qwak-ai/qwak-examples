import os

import numpy as np
import pandas as pd
import qwak
from catboost import CatBoostClassifier, Pool, cv
from catboost.datasets import titanic
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, InferenceOutput, ModelSchema
from sklearn.model_selection import train_test_split


class TitanicSurvivalPrediction(QwakModel):
    def __init__(self):
        loss_function = os.getenv("loss_fn", "Logloss")
        iterations = int(os.getenv("iterations", 1000))
        learning_rate = os.getenv("learning_rate", None)
        if learning_rate:
            learning_rate = int(learning_rate)

        custom_loss = "Accuracy"
        self.model = CatBoostClassifier(
            iterations=iterations,
            custom_loss=[custom_loss],
            loss_function=loss_function,
            learning_rate=learning_rate,
        )

        qwak.log_param({
            "loss_function": loss_function,
            "learning_rate": learning_rate,
            "iterations": iterations,
            "custom_loss": custom_loss,
        })

    def build(self):
        titanic_train, _ = titanic()

        # for the train data ,the age ,fare and embarked has null value,so just make it -999 for it
        # and the catboost will distinguish it
        titanic_train.fillna(-999, inplace=True)

        x = titanic_train.drop(["Survived", "PassengerId"], axis=1)
        y = titanic_train.Survived

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=0.85, random_state=42
        )

        # mark categorical features
        cate_features_index = np.where(x_train.dtypes != float)[0]

        self.model.fit(
            x_train,
            y_train,
            cat_features=cate_features_index,
            eval_set=(x_test, y_test),
        )

        # Cross validating the model (2-fold)
        cv_data = cv(
            Pool(x, y, cat_features=cate_features_index),
            self.model.get_params(),
            fold_count=2,
        )
        print(
            "the best cross validation accuracy is :{}".format(
                np.max(cv_data["test-Accuracy-mean"])
            )
        )
        qwak.log_metric({"val_accuracy": np.max(cv_data["test-Accuracy-mean"])})

    def schema(self):
        return ModelSchema(
            inputs=[
                ExplicitFeature(name="PassengerId", type=int),
                ExplicitFeature(name="Pclass", type=int),
                ExplicitFeature(name="Name", type=str),
                ExplicitFeature(name="Sex", type=str),
                ExplicitFeature(name="Age", type=int),
                ExplicitFeature(name="SibSp", type=int),
                ExplicitFeature(name="Parch", type=int),
                ExplicitFeature(name="Ticket", type=str),
                ExplicitFeature(name="Fare", type=float),
                ExplicitFeature(name="Cabin", type=str),
                ExplicitFeature(name="Embarked", type=str),
            ],
            outputs=[InferenceOutput(name="Survived_Probability", type=float)],
        )

    @qwak.api()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(["PassengerId"], axis=1)

        # Fill missing values in categorical features
        default_values = {
            "Sex": "Unknown",
            "Ticket": "Unknown",
            "Cabin": "Unknown",
            "Embarked": "Unknown",
            "Name": "Unknown"
        }

        # Fill missing values in the DataFrame with default values
        df.fillna(default_values, inplace=True)

        return pd.DataFrame(
                self.model.predict_proba(df[self.model.feature_names_])[:, 1],
                columns=['Survived_Probability']
            )
