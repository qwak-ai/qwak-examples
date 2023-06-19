import os

import catboost
import numpy as np
import pandas as pd
import qwak
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, ModelSchema, InferenceOutput
from sklearn.model_selection import train_test_split


class CatBoostCreditRiskModel(QwakModel):

    def __init__(self):
        self.params = {
            'iterations': int(os.getenv('iterations', 1000)),
            'learning_rate': float(os.getenv('learning_rate', 0.1)),
            'random_seed': int(os.getenv('random_seed', 7)),
            'loss_function': os.getenv('loss_fn', 'Logloss'),
            'eval_metric': 'Accuracy',
            'logging_level': 'Silent',
            'use_best_model': True
        }

        # A CatBoost classifier with the specified parameters
        self.model = catboost.CatBoostClassifier(**self.params)

        # Log model parameters to Qwak for tracking purposes
        qwak.log_param(self.params)

    def build(self):
        """
        The build() method is called once during the build process.
        Use it for training or actions during the model build phase.
        """

        # Load the credit risk dataset
        file_absolute_path = os.path.dirname(os.path.abspath(__file__))
        df_credit = pd.read_csv(f'{file_absolute_path}/data.csv', index_col=0)

        # Create a categorical variable to handle the "Age Category"
        interval = (18, 25, 35, 60, 120)
        categories = ['Student', 'Young', 'Adult', 'Senior']
        df_credit["Age_cat"] = pd.cut(
            df_credit.Age,
            interval,
            labels=categories
        ).astype(object)

        # Fill in the missing values in the fields below
        df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna('no_inf')
        df_credit['Checking account'] = df_credit['Checking account'].fillna('no_inf')

        df_credit = df_credit.merge(
            pd.get_dummies(df_credit.Risk, prefix='Risk'),
            left_index=True,
            right_index=True
        )

        # Excluding the missing columns
        del df_credit['Risk']
        del df_credit['Risk_good']

        df_credit['Credit amount'] = np.log(df_credit['Credit amount'])

        # Creating the X and y variables
        X = df_credit.drop(['Risk_bad'], axis=1)
        y = df_credit['Risk_bad']
        categorical_features_indices = np.where(X.dtypes == object)[0]

        # Splitting X and y into train and test version
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=self.params['random_seed']
        )

        # Training our CatBoost model
        self.model.fit(
            X_train,
            y_train,
            cat_features=categorical_features_indices,
            eval_set=(X_test, y_test)
        )

        # Cross validating the model (5-fold)
        cv_data = catboost.cv(
            catboost.Pool(
                X_train,
                y_train,
                cat_features=categorical_features_indices
            ),
            self.model.get_params(),
            fold_count=5
        )

        max_accuracy = np.max(cv_data["test-Accuracy-mean"])
        print(f'Best cross validation accuracy:{max_accuracy}')
        qwak.log_metric({"val_accuracy": max_accuracy})

    def schema(self):
        """
        schema() define the model input structure, and is used to enforce 
        the correct structure of incoming prediction requests.
        """
        return ModelSchema(
            inputs=[
                ExplicitFeature(name='UserId', type=str),
                ExplicitFeature(name='Age', type=int),
                ExplicitFeature(name='Sex', type=str),
                ExplicitFeature(name='Job', type=int),
                ExplicitFeature(name='Housing', type=str),
                ExplicitFeature(name='Saving accounts', type=str),
                ExplicitFeature(name='Checking account', type=str),
                ExplicitFeature(name='Credit amount', type=float),
                ExplicitFeature(name='Duration', type=int),
                ExplicitFeature(name='Purpose', type=str),
                ExplicitFeature(name='Age_cat', type=str)
            ],
            outputs=[
                InferenceOutput(name='Default_Probability', type=float)
            ])

    @qwak.api()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The predict(df) method is the live inference method 
        """
        df = df.drop(['UserId'], axis=1)
        return pd.DataFrame(
            self.model.predict_proba(df[self.model.feature_names_])[:, 1],
            columns=['Default_Probability']
        )
