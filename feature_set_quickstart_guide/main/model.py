# Importing the FrogMlModel interface
from frogml import FrogMlModel

# Importing the Feature Store clients used to fetch results
from frogml.feature_store.offline import OfflineClientV2
from frogml.core.feature_store.offline.feature_set_features import FeatureSetFeatures
from datetime import datetime

# Importing the Features schema and prediction adapters
from frogml.sdk.model.schema import ModelSchema, InferenceOutput
from frogml.sdk.model.schema_entities import FeatureStoreInput

from catboost import cv, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

from datetime import date
import pandas as pd
import numpy as np
import frogml

from main.utils import features_cleaning

FEATURE_SET = "user-credit-risk-features"
ENTITY_KEY = "user_id"


# CreditRiskModel class definition, inheriting from FrogMlModel
class CreditRiskModel(FrogMlModel):

    # Class constructor - anything initialized here will be `pickled` with the Docker Image
    def __init__(self):
        # Initializing CatBoost Regressor with specific hyperparameters
        # This is the model that will be trained and used for predictions
        self.model = CatBoostRegressor(
            iterations=10,
            loss_function='RMSE',
            learning_rate=None
        )

        # Define the date range for data retrieval
        self.feature_range_start = date(2020, 1, 1)
        self.feature_range_end = date.today()

        # Logging training set date range for tracking and reproducibility
        # The parameters will also be available in the JFrogML UI for each Model Build
        frogml.log_param({"features_start_range": self.feature_range_start,
                   "features_end_range": self.feature_range_end})

    # Method called by the JFrogML Cloud to train and build the model
    def build(self):
        # Define the features to be used for the model and fetched from the Offline Feature Store
        # These are the specific features that the model will be trained on

        offline_feature_store = OfflineClientV2()
        features = FeatureSetFeatures(feature_set_name='user-credit-risk-features',
                                      feature_names=['checking_account', 'age', 'job', 'duration',
                                                   'credit_amount', 'housing', 'purpose', 'saving_account', 'sex'])

        # Fetch data from the offline client
        data = offline_feature_store.get_feature_range_values(
            features=features,
            start_date=datetime(year=2021, month=1, day=1),
            end_date=datetime(year=2021, month=1, day=3)
        )

        # Logging hyperparameters for tracking and reproducibility
        # The parameters will also be available in the JFrogML UI for each Model Build
        params = self.model.get_params()
        frogml.log_param({"iterations": params['iterations'], "loss_function": params['loss_function']})

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

        # Log the metrics to JFrogML to have a basis of performance comparison between Model Builds
        frogml.log_metric(
            {
                "val_rmse_mean": max_mean_row["test-RMSE-mean"][0],
                "val_rmse_std": max_mean_row["test-RMSE-std"][0],
            }
        )

    # Define the schema for the Model and Feature Store
    # This tells JFrogML how to deserialize the output of the Prediction method as well as what
    # features to retrieve from the Online Feature Store for inference without explicitly specifying every time.
    def schema(self) -> ModelSchema:
        model_schema = ModelSchema(inputs=[
            FeatureStoreInput(name=f'{FEATURE_SET}.checking_account'),
            FeatureStoreInput(name=f'{FEATURE_SET}.age'),
            FeatureStoreInput(name=f'{FEATURE_SET}.job'),
            FeatureStoreInput(name=f'{FEATURE_SET}.duration'),
            FeatureStoreInput(name=f'{FEATURE_SET}.credit_amount'),
            FeatureStoreInput(name=f'{FEATURE_SET}.housing'),
            FeatureStoreInput(name=f'{FEATURE_SET}.purpose'),
            FeatureStoreInput(name=f'{FEATURE_SET}.saving_account'),
            FeatureStoreInput(name=f'{FEATURE_SET}.sex'),
        ],
            outputs=[InferenceOutput(name="score", type=float)])
        return model_schema

    # The JFrogML API decorator wraps the predict function with additional functionality and wires additional dependencies.
    # This allows external services to call this method for making predictions.

    @frogml.api(feature_extraction=True)
    def predict(self, df: pd.DataFrame, extracted_df: pd.DataFrame) -> pd.DataFrame:
        # Prediction method that takes a DataFrame with the User IDs as input, enriches it with Features and returns predictions

        # Cleaning the features to prepare them for inference
        X, y = features_cleaning(extracted_df)

        print("Retrieved the following features from the Online Feature Store:\n\n", X)

        # Calling the model prediction function and converting the NdArray to a List to be serializable as JSON
        prediction = self.model.predict(X).tolist()

        return prediction
