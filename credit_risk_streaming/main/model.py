import numpy as np
from qwak.feature_store.offline import OfflineClientV2
from qwak.feature_store.offline.feature_set_features import FeatureSetFeatures
import datetime
import pandas as pd
import qwak
from qwak.model.base import QwakModel
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
import pandas as pd
from qwak.model.tools import run_local



import os

RUNNING_FILE_ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


class StreamingRiskModel(QwakModel):

    def __init__(self):
        self.params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'eval_metric': 'Accuracy',
            'logging_level': 'Silent',
            'use_best_model': True
        }
        self.catboost = CatBoostClassifier(**self.params)
        self.metrics = {
            'accuracy': 90,
            'random_state': 42,
            'test_size': .25
        }
        qwak.log_param(self.params)


    def fetch_features(self):
        """
        Read data from the offline feature store
        :return: Feature Store DF
        """
        print("Fetching data from the feature store")
        offline_feature_store = OfflineClientV2()
        population_df = pd.read_csv(f"{RUNNING_FILE_ABSOLUTE_PATH}/population.csv")

        streaming_features = FeatureSetFeatures(
            feature_set_name='transaction-aggregates-demo',
            feature_names=['count_transaction_amount_1m',
                        'last_distinct_5_transaction_amount_1m',
                        'sum_transaction_amount_1m',
                        'sample_stdev_transaction_amount_1m',
                        'median_transaction_amount_1m',
                        'max_transaction_amount_1m',
                        'count_transaction_amount_1h',
                        'sum_transaction_amount_1h',
                        'sample_stdev_transaction_amount_1h',
                        'median_transaction_amount_1h',
                        'max_transaction_amount_1h'
                        ]
        )

        batch_features = FeatureSetFeatures(
            feature_set_name='qwak-snowflake-webinar',
            feature_names=['job','credit_amount','duration','purpose','risk']
        )
        features = [streaming_features, batch_features]
        return offline_feature_store.get_feature_values(
            features=features,
            population=population_df
        )

    def build(self):
        """
        Build the Qwak model:
            1. Fetch the feature values from the feature store
            2. Train a naive Catboost model
        """
        df = self.fetch_features()
        print(df.columns)
        train_df = df[["qwak-snowflake-webinar.job", "qwak-snowflake-webinar.credit_amount", "qwak-snowflake-webinar.duration", "qwak-snowflake-webinar.purpose","transaction-aggregates-demo.count_transaction_amount_1m","transaction-aggregates-demo.sum_transaction_amount_1m","transaction-aggregates-demo.sample_stdev_transaction_amount_1m","transaction-aggregates-demo.median_transaction_amount_1m","transaction-aggregates-demo.max_transaction_amount_1m","transaction-aggregates-demo.count_transaction_amount_1h","transaction-aggregates-demo.sum_transaction_amount_1h","transaction-aggregates-demo.sample_stdev_transaction_amount_1h","transaction-aggregates-demo.median_transaction_amount_1h","transaction-aggregates-demo.max_transaction_amount_1h" ]]

        y = df["qwak-snowflake-webinar.risk"].map({'good':1,'bad':0})


        categorical_features_indices = np.where(train_df.dtypes != np.float64)[0]
        X_train, X_validation, y_train, y_validation = train_test_split(train_df, y, test_size=0.25, random_state=42)

        train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
        validate_pool = Pool(X_validation, y_validation, cat_features=categorical_features_indices)

        print("Fitting catboost model")
        self.catboost.fit(train_pool, eval_set=validate_pool)

        y_predicted = self.catboost.predict(X_validation)
        f1 = f1_score(y_validation, y_predicted)
        
        qwak.log_metric({'f1_score': f1})
        qwak.log_metric({'iterations': self.params['iterations']})
        qwak.log_metric({'learning_rate': self.params['learning_rate']})
        qwak.log_metric({'accuracy': self.metrics['accuracy']})
        qwak.log_metric({'random_state': self.metrics['random_state']})
        qwak.log_metric({'test_size': self.metrics['test_size']})
        print("************************ finished")
        
    
        




    def schema(self):
        from qwak.model.schema import ModelSchema, InferenceOutput, FeatureStoreInput, Entity
        user_id = Entity(name="user_id", type=str)
        model_schema = ModelSchema(
            entities=[user_id],
            inputs=[
                FeatureStoreInput(entity=user_id, name="transaction-aggregates-demo.count_transaction_amount_1m"),
                FeatureStoreInput(entity=user_id, name="transaction-aggregates-demo.sum_transaction_amount_1m"),
                FeatureStoreInput(entity=user_id, name="transaction-aggregates-demo.sample_stdev_transaction_amount_1m"),
                FeatureStoreInput(entity=user_id, name="transaction-aggregates-demo.median_transaction_amount_1m"),
                FeatureStoreInput(entity=user_id, name="transaction-aggregates-demo.max_transaction_amount_1m"),
                FeatureStoreInput(entity=user_id, name="transaction-aggregates-demo.count_transaction_amount_1h"),
                FeatureStoreInput(entity=user_id, name="transaction-aggregates-demo.sum_transaction_amount_1h"),
                FeatureStoreInput(entity=user_id, name="transaction-aggregates-demo.sample_stdev_transaction_amount_1h"),
                FeatureStoreInput(entity=user_id, name="transaction-aggregates-demo.median_transaction_amount_1h"),
                FeatureStoreInput(entity=user_id, name="transaction-aggregates-demo.max_transaction_amount_1h"),
                FeatureStoreInput(entity=user_id, name='qwak-snowflake-webinar.job'),
                FeatureStoreInput(entity=user_id, name='qwak-snowflake-webinar.credit_amount'),
                FeatureStoreInput(entity=user_id, name='qwak-snowflake-webinar.duration'),
                FeatureStoreInput(entity=user_id, name='qwak-snowflake-webinar.purpose'),

            ],
            outputs=[
                InferenceOutput(name="Risk", type=float)
            ])
        return model_schema

    @qwak.api(feature_extraction=True)
    def predict(self, df, extracted_df):
        #### {"user_id": "xxxx-xxx-xxx-xxxx"}
        # clean_df = self.clean_df(extracted_df)
        return pd.DataFrame(self.catboost.predict(extracted_df[["qwak-snowflake-webinar.job", "qwak-snowflake-webinar.credit_amount", "qwak-snowflake-webinar.duration", "qwak-snowflake-webinar.purpose","transaction-aggregates-demo.count_transaction_amount_1m","transaction-aggregates-demo.sum_transaction_amount_1m","transaction-aggregates-demo.sample_stdev_transaction_amount_1m","transaction-aggregates-demo.median_transaction_amount_1m","transaction-aggregates-demo.max_transaction_amount_1m","transaction-aggregates-demo.count_transaction_amount_1h","transaction-aggregates-demo.sum_transaction_amount_1h","transaction-aggregates-demo.sample_stdev_transaction_amount_1h","transaction-aggregates-demo.median_transaction_amount_1h","transaction-aggregates-demo.max_transaction_amount_1h" ]]),
                            columns=['Risk'])
    
    def clean_df(self, df): 
        clean = df.rename(columns={
            "qwak-snowflake-webinar.job": "job",
            "qwak-snowflake-webinar.credit_amount": "credit_amount", 
            "qwak-snowflake-webinar.duration": "duration",
            "qwak-snowflake-webinar.purpose": "purpose",
            "transaction-aggregates-demo.count_transaction_amount_1m": "count_transaction_amount_1m",
            "transaction-aggregates-demo.sum_transaction_amount_1m": "sum_transaction_amount_1m",
            "transaction-aggregates-demo.sample_stdev_transaction_amount_1m": "sample_stdev_transaction_amount_1m",
            "transaction-aggregates-demo.median_transaction_amount_1m": "median_transaction_amount_1m",
            "transaction-aggregates-demo.max_transaction_amount_1m":   "max_transaction_amount_1m",
            "transaction-aggregates-demo.count_transaction_amount_1h": "count_transaction_amount_1h",
            "transaction-aggregates-demo.sum_transaction_amount_1h": "sum_transaction_amount_1h",
            "transaction-aggregates-demo.sample_stdev_transaction_amount_1h": "sample_stdev_transaction_amount_1h",
            "transaction-aggregates-demo.median_transaction_amount_1h": "median_transaction_amount_1h",
            "transaction-aggregates-demo.max_transaction_amount_1h":"max_transaction_amount_1h"
            })
        return clean

# if __name__ == '__main__':
#     # Create a new instance of the model
#     m = StreamingRiskModel()
    
#     # Create an input vector and convert it to JSON
#     input_vector = pd.DataFrame(
#         [{
#             "user_id": "b0ca3ac4-5432-4c21-8251-a6ae0d3ad874"
#         }]
#     ).to_json()
    
#     # Run local inference using the model
#     prediction = run_local(m, input_vector)
#     print(prediction)