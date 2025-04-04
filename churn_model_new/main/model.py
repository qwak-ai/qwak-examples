import os

import pandas as pd
import qwak
import xgboost as xgb
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, ModelSchema, InferenceOutput
from sklearn.model_selection import train_test_split
from qwak.tools.logger import get_qwak_logger
import logging 
import io
from joblib import dump

logger = get_qwak_logger()

class XGBoostChurnPredictionModel(QwakModel):

    def __init__(self):
        self.params = {
            'n_estimators': int(os.getenv('n_estimators', 300)),
            'learning_rate': float(os.getenv('learning_rate', 0.05)),
            'objective': 'binary:logistic'
        }

        # Create a XGBoost classifier with the specified parameters
        self.model = xgb.XGBClassifier(**self.params)

        # Log model parameters to Qwak for tracking purposes
        qwak.log_param(self.params)


    def configure_logger(self):
        # Create a logger
        logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

        # Create a StringIO buffer to capture log messages
        log_buffer = io.StringIO()

        # Create a StreamHandler with the buffer
        stream_handler = logging.StreamHandler(log_buffer)
        stream_handler.setLevel(logging.CRITICAL)  # Set level to CRITICAL to filter messages

        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)

        # Add a filter to the handler to capture only CRITICAL messages
        class CriticalFilter(logging.Filter):
            def filter(self, record):
                return record.levelno == logging.CRITICAL

        stream_handler.addFilter(CriticalFilter())

        # Add the handler to the logger
        logger.addHandler(stream_handler)

        return logger, log_buffer

    def build(self):


        # Configure the logger
        logger, log_buffer = self.configure_logger()

        # Log some messages
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        logger.critical("This is a critical message")
        logger.critical("Another critical message")

        # Retrieve critical messages from the log buffer
        log_buffer.seek(0)  # Rewind the buffer to the beginning
        critical_logs = log_buffer.read().strip().split('\n')  # Split log messages into a list

        # Print captured critical messages
        print("\nCritical Messages:")
        for log_message in critical_logs:
            print(log_message)

        # Close the buffer
        log_buffer.close()

        file_absolute_path = os.path.dirname(os.path.abspath(__file__))
        df = pd.read_csv(f"{file_absolute_path}/data.csv")

        # Creating the X and y variables
        y = df['churn']
        X = df.drop(['churn', 'User_Id', '__index_level_0__',
                     'event date', 'Phone', 'State'], axis=1)

        # Splitting X and y into train and test version
        X_train, X_validation, y_train, y_validation = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # Training our CatBoost model
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_validation, y_validation)]
        )

        dump(self.model, "churn_model.joblib")

        # Log metrics into Qwak
        accuracy = self.model.score(X_validation, y_validation)
        qwak.log_metric({"val_accuracy": accuracy})
        qwak.log_data(dataframe=X, tag="train_data")

    @qwak.api()
    def predict(self, df):
        """
        The predict(df) method is the actual inference method.
        """
        # Getting the original columns order
        feature_order = self.model.get_booster().feature_names

        # Reformatting the prediction data order
        prediction_data = df.drop(
            ['User_Id', 'State'], axis=1
        ).reindex(columns=feature_order)

        predictions = self.model.predict(prediction_data)

        return pd.DataFrame(
            predictions,
            columns=['Churn_Probability']
        )

    def schema(self):
        """
        schema() define the model input structure.
        Use it to enforce the structure of incoming requests.
        """
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="User_Id", type=str),
                ExplicitFeature(name="State", type=str),
                ExplicitFeature(name="Account_Length", type=int),
                ExplicitFeature(name="Area_Code", type=str),
                ExplicitFeature(name="Intl_Plan", type=int),
                ExplicitFeature(name="VMail_Plan", type=int),
                ExplicitFeature(name="VMail_Message", type=int),
                ExplicitFeature(name="Day_Mins", type=float),
                ExplicitFeature(name="Day_Calls", type=int),
                ExplicitFeature(name="Eve_Mins", type=float),
                ExplicitFeature(name="Eve_Calls", type=int),
                ExplicitFeature(name="Night_Mins", type=float),
                ExplicitFeature(name="Night_Calls", type=int),
                ExplicitFeature(name="Intl_Mins", type=float),
                ExplicitFeature(name="Intl_Calls", type=int),
                ExplicitFeature(name="CustServ_Calls", type=int),
                ExplicitFeature(name="Agitation_Level", type=int),
            ],
            outputs=[
                InferenceOutput(name="Churn_Probability", type=float)
            ])
        return model_schema
