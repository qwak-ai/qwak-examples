import os
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    params = {
        'iterations': int(os.getenv('iterations', 500)),
        'learning_rate': float(os.getenv('learning_rate', 0.1)),
        'eval_metric': 'Accuracy',
        'random_seed': int(os.getenv('random_seed', 42)),
        'logging_level': 'Silent',
        'use_best_model': True
    }
    catboost = CatBoostClassifier(**params)

    df = pd.read_csv("./data.csv")
    y = df['churn']
    X = df.drop(['churn', 'User_Id'], axis=1)

    categorical_features_indices = np.where(X.dtypes != np.float64)[0]
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.25, random_state=42)

    train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
    validate_pool = Pool(X_validation, y_validation, cat_features=categorical_features_indices)

    catboost.fit(train_pool, eval_set=validate_pool)
    catboost.save_model('./main/model.cbm')
