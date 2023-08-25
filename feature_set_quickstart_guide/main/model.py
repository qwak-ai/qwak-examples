from qwak.model.base import QwakModel
from catboost import cv, CatBoostRegressor, Pool
from qwak.feature_store.offline.client import OfflineClient
from qwak.feature_store.online.client import OnlineClient
from qwak.model.schema import ModelSchema, InferenceOutput
from qwak.model.schema_entities import FeatureStoreInput
from sklearn.model_selection import train_test_split
from qwak import log_param, log_metric
from datetime import date
#from multiprocessing import Pool
from typing import Tuple
import pandas as pd
import numpy as np
import qwak
from main.feature_set import entity

# CreditRiskModel class definition, inheriting from QwakModel
class CreditRiskModel(QwakModel):
   
    def __init__(self):

        # Initializing CatBoost Regressor with specific hyperparameters
        # This is the model that will be trained and used for predictions
        self.model = CatBoostRegressor(
            iterations=1000,
            loss_function='RMSE',
            learning_rate=None
        )

        """
        Creating an OnlineClient once, to fetch features every time during prediction. 
        This object will be serialized along with the model class and re-created after deployment.
        """
        self.online_client = OnlineClient()


    def build(self):

        # Define the features to be used for the model and fetched from the Offline Feature Store
        # These are the specific features that the model will be trained on
        key_to_features = {'user': [
            'user-credit-risk-features.checking_account',
            'user-credit-risk-features.age',
            'user-credit-risk-features.job',
            'user-credit-risk-features.duration',
            'user-credit-risk-features.credit_amount',
            'user-credit-risk-features.housing',
            'user-credit-risk-features.purpose',
            'user-credit-risk-features.saving_account',
            'user-credit-risk-features.sex'
            ]
        }

        # Creating an OfflineClient to fetch historical feature data
        offline_client = OfflineClient()

        # Define the date range for data retrieval
        start_date = date(2020, 1, 1)
        end_date = date.today()

        # Fetch data from the offline client
        data = offline_client.get_feature_range_values(
            entity_key_to_features=key_to_features,
            start_date=start_date,
            end_date=end_date
        )

        # Logging hyperparameters for tracking and reproducibility
        # The parameters will also be available in the Qwak UI for each Model Build
        params = self.model.get_params()
        log_param({"iterations": params['iterations'], "loss_function": params['loss_function']})

        # Clean and split the features
        X, y = self.features_cleaning(data)
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=42)
        cate_features_index = np.where(x_train.dtypes != int)[0]

        # Train the model
        self.model.fit(x_train, 
                       y_train, 
                       cat_features=cate_features_index, 
                       eval_set=(x_test, y_test)
        )


        # Create a Pool object with the categorical features
        pool_data = Pool(X, y, cat_features=cate_features_index)

        # Cross-validation
        cv_data = cv(pool_data, 
                     self.model.get_params(), 
                     fold_count=5)

        max_mean_row = cv_data[cv_data["test-RMSE-mean"] == np.max(cv_data["test-RMSE-mean"])]

        # Log the metrics to Qwak to have a basis of performance comparison between Model Builds
        log_metric(
            {
                "val_rmse_mean": max_mean_row["test-RMSE-mean"][0],
                "val_rmse_std": max_mean_row["test-RMSE-std"][0],
            }
        )


    # Define the schema for the Model and Feature Store
    # This tells Qwak how to deserialize the output of the Predictiom method as well as what 
    # features to retrieve from the Online Feature Store for inference without explicitly specifying every time.
    def schema(self) -> ModelSchema:

        model_schema = ModelSchema(entities=[entity],
                                   inputs=[
                                        FeatureStoreInput(name='user-credit-risk-features.checking_account', entity=entity),
                                        FeatureStoreInput(name='user-credit-risk-features.age', entity=entity),
                                        FeatureStoreInput(name='user-credit-risk-features.job', entity=entity),
                                        FeatureStoreInput(name='user-credit-risk-features.duration', entity=entity),
                                        FeatureStoreInput(name='user-credit-risk-features.credit_amount', entity=entity),
                                        FeatureStoreInput(name='user-credit-risk-features.housing', entity=entity),
                                        FeatureStoreInput(name='user-credit-risk-features.purpose', entity=entity),
                                        FeatureStoreInput(name='user-credit-risk-features.saving_account', entity=entity),
                                        FeatureStoreInput(name='user-credit-risk-features.sex', entity=entity),
                                    ],
                                    outputs=[InferenceOutput(name="score", type=float)])
        return model_schema


    # The Qwak API decorator wraps the predict function with additional functionality and wires additional adependencies. 
    # This allows external services to call this method for making predictions.
    @qwak.api()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # Prediction method that takes a DataFrame with the User IDs as input and returns predictions
        
        # Calling the OnlineClient to retrieve the features for the given User ID
        features = self.online_client.get_feature_values(self.schema(), df)

        # Cleaning the features to prepare them for inference
        X, y = self.features_cleaning(features)

        print("Retrieved the following features from the Online Feature Store:\n\n", X)

        # Calling the model prediction function and converting the NdArray to a List to be serializable as JSON
        prediction = self.model.predict(X).tolist()

        return prediction



    # Utility function
    def features_cleaning(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Method to clean and prepare the features for training or prediction
        # This includes renaming columns, selecting specific features, and handling missing values
        
        data = data.rename(
            columns={
                "user-credit-risk-features.checking_account": "checking_account",
                "user-credit-risk-features.age": "age",
                "user-credit-risk-features.job": "job",
                "user-credit-risk-features.duration": "duration",
                "user-credit-risk-features.credit_amount": "credit_amount",
                "user-credit-risk-features.housing": "housing",
                "user-credit-risk-features.purpose": "purpose",
                "user-credit-risk-features.saving_account": "saving_account",
                "user-credit-risk-features.sex": "sex",
            }
        )

        data = data[
            [
                "checking_account",
                "age",
                "job",
                "credit_amount",
                "housing",
                "purpose",
                "saving_account",
                "sex",
                "duration",
            ]
        ]
        data = data.dropna()  # in production, we should fill the missing values
        # but we don't have a second data source for the missing data, so let's drop them

        X = data[
            [
                "checking_account",
                "age",
                "job",
                "credit_amount",
                "housing",
                "purpose",
                "saving_account",
                "sex",
            ]
        ]
        y = data[["duration"]]

        return X, y