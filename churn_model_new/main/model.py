import os
import pandas as pd
import frogml
import xgboost as xgb
from frogml import FrogMlModel
from frogml.sdk.model.schema import ExplicitFeature, ModelSchema, InferenceOutput
from sklearn.model_selection import train_test_split


class XGBoostChurnPredictionModel(FrogMlModel):

    def __init__(self):
        self.params = {
            'n_estimators': int(os.getenv('n_estimators', 300)),
            'learning_rate': float(os.getenv('learning_rate', 0.05)),
            'objective': 'binary:logistic'
        }

        # Create a XGBoost classifier with the specified parameters
        self.model = xgb.XGBClassifier(**self.params)

        # Log model parameters to Qwak for tracking purposes
        frogml.log_param(self.params)

    def build(self):
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

        # Log metrics into Qwak
        accuracy = self.model.score(X_validation, y_validation)
        frogml.log_metric({"val_accuracy": accuracy})
        frogml.log_data(dataframe=X, tag="train_data")

    @frogml.api()
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
