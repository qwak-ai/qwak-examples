from qwak.feature_store.offline import OfflineFeatureStore
from multiprocessing import Pool
import qwak
from qwak import log_param, log_metric
from qwak.model import QwakModelInterface
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool, cv
import numpy as np

class CreditRisk(QwakModelInterface):

    def __init__(self):
        self.model = CatBoostRegressor(
            iterations=1000,
            loss_function='RMSE',
            learning_rate=None
        )

    def build(self):
        offline_feature_store = OfflineFeatureStore()
        data = offline_feature_store.get_sample_data(feature_set_name='user_credit_risk_features_v2',
                                                     number_of_rows=999)

        log_param({'iterations': 1000, 'loss_function': 'RMSE'})

        data = data.rename(columns={
            'user_credit_risk_features_v2.checking_account': 'checking_account',
            'user_credit_risk_features_v2.age': 'age',
            'user_credit_risk_features_v2.job': 'job',
            'user_credit_risk_features_v2.duration': 'duration',
            'user_credit_risk_features_v2.credit_amount': 'credit_amount',
            'user_credit_risk_features_v2.housing': 'housing',
            'user_credit_risk_features_v2.purpose': 'purpose',
            'user_credit_risk_features_v2.saving_account': 'saving_account',
            'user_credit_risk_features_v2.sex': 'sex'
        })

        data = data[['checking_account', 'age', 'job', 'credit_amount', 'housing', 'purpose', 'saving_account', 'sex',
                     'duration']]
        data = data.dropna()  # in production, we should fill the missing values
        # but we don't have a second data source for the missing data, so let's drop them

        x = data[['checking_account', 'age', 'job', 'credit_amount', 'housing', 'purpose', 'saving_account', 'sex']]
        y = data[['duration']]

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.85, random_state=42)
        cate_features_index = np.where(x_train.dtypes != int)[0]
        self.model.fit(x_train, y_train, cat_features=cate_features_index, eval_set=(x_test, y_test))

        cv_data = cv(Pool(x, y, cat_features=cate_features_index), self.model.get_params(), fold_count=5)
        max_mean_row = cv_data[cv_data['test-RMSE-mean'] == np.max(cv_data["test-RMSE-mean"])]
        log_metric(
            {"val_rmse_mean": max_mean_row["test-RMSE-mean"][0], 'val_rmse_std': max_mean_row["test-RMSE-std"][0]})

    def schema(self):
        from qwak.model.schema import ModelSchema, FeatureStoreInput, Prediction
        model_schema = ModelSchema(
            features=[
                FeatureStoreInput(name='user_credit_risk_features_v2.checking_account'),
                FeatureStoreInput(name='user_credit_risk_features_v2.age'),
                FeatureStoreInput(name='user_credit_risk_features_v2.job'),
                FeatureStoreInput(name='user_credit_risk_features_v2.credit_amount'),
                FeatureStoreInput(name='user_credit_risk_features_v2.housing'),
                FeatureStoreInput(name='user_credit_risk_features_v2.purpose'),
                FeatureStoreInput(name='user_credit_risk_features_v2.saving_account'),
                FeatureStoreInput(name='user_credit_risk_features_v2.sex'),
                FeatureStoreInput(name='user_credit_risk_features_v2.duration'),
            ],
            predictions=[
                Prediction(name="duration", type=float)
            ])
        return model_schema

    @qwak.api(feature_extraction=True)
    def predict(self, df, extracted_df):
        return pd.DataFrame(self.model.predict(extracted_df))


if __name__ == '__main__':
    model = CreditRisk()
    model.build()

    import pandas as pd

    feature_vector = pd.DataFrame([{
        "user_id": "8b65e705-bd8e-4859-a72f-851998eb5688"
    }])

    print(model.predict(feature_vector, None))
