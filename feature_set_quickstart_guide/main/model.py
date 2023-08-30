
# Importing the QwakModel interface
from qwak.model.base import QwakModel

# Importing the Feature Store clients used to fetch results
from qwak.feature_store.offline.client import OfflineClient
from qwak.feature_store.online.client import OnlineClient

# Importing the Features schema and prediction adapters
from qwak.model.schema import ModelSchema, InferenceOutput
from qwak.model.schema_entities import FeatureStoreInput

from catboost import cv, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

# Utility methods to log metrics and model parameters to Qwak Cloud
from qwak import log_param, log_metric

from datetime import date
import pandas as pd
import numpy as np
import qwak

from main.utils import features_cleaning
from main.feature_set import entity

# CreditRiskModel class definition, inheriting from QwakModel
class CreditRiskModel(QwakModel):
   
   # Class constructor - anything initialized here will be `pickled` with the Docker Image
    def __init__(self):

        # Initializing CatBoost Regressor with specific hyperparameters
        # This is the model that will be trained and used for predictions
        self.model = CatBoostRegressor(
            iterations=1000,
            loss_function='RMSE',
            learning_rate=None
        )

        # Define the date range for data retrieval
        self.feature_range_start = date(2020, 1, 1)
        self.feature_range_end = date.today()

        # Logging training set date range for tracking and reproducibility
        # The parameters will also be available in the Qwak UI for each Model Build
        log_param({"features_start_range": self.feature_range_start, 
                   "features_end_range": self.feature_range_end})


    # Method called by the Qwak Cloud to train and build the model
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

        offline_client = OfflineClient()

        # Fetch data from the offline client
        data = offline_client.get_feature_range_values(
            entity_key_to_features=key_to_features,
            start_date=self.feature_range_start,
            end_date=self.feature_range_end
        )

        # Logging hyperparameters for tracking and reproducibility
        # The parameters will also be available in the Qwak UI for each Model Build
        params = self.model.get_params()
        log_param({"iterations": params['iterations'], "loss_function": params['loss_function']})

        # Clean and split the features
        X, y = features_cleaning(data)
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

    # Method called at deployment to initialize the model before it starts predicting
    def initialize_model(self):

        """
        Creating an OnlineClient once, to fetch features every time during prediction. 
        This object will be serialized along with the model class and re-created after deployment.
        """
        self.online_client = OnlineClient()

        return super().initialize_model()


    # Define the schema for the Model and Feature Store
    # This tells Qwak how to deserialize the output of the Predictiom method as well as what 
    # features to retrieve from the Online Feature Store for inference without explicitly specifying every time.
    def schema(self) -> ModelSchema:

        model_schema = ModelSchema(inputs=[
                                        FeatureStoreInput(name='user-credit-risk-features.checking_account'),
                                        FeatureStoreInput(name='user-credit-risk-features.age'),
                                        FeatureStoreInput(name='user-credit-risk-features.job'),
                                        FeatureStoreInput(name='user-credit-risk-features.duration'),
                                        FeatureStoreInput(name='user-credit-risk-features.credit_amount'),
                                        FeatureStoreInput(name='user-credit-risk-features.housing'),
                                        FeatureStoreInput(name='user-credit-risk-features.purpose'),
                                        FeatureStoreInput(name='user-credit-risk-features.saving_account'),
                                        FeatureStoreInput(name='user-credit-risk-features.sex'),
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
        X, y = features_cleaning(features)

        print("Retrieved the following features from the Online Feature Store:\n\n", X)

        # Calling the model prediction function and converting the NdArray to a List to be serializable as JSON
        prediction = self.model.predict(X).tolist()

        return prediction
