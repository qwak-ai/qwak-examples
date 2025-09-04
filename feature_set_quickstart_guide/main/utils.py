from typing import Tuple
import pandas as pd

FEATURE_SET = "user-credit-risk-features"

# Utility function
def features_cleaning(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Method to clean and prepare the features for training or prediction
    # This includes renaming columns, selecting specific features, and handling missing values
    
    data = data.rename(
        columns={
            f'{FEATURE_SET}.checking_account': "checking_account",
            f'{FEATURE_SET}.age': "age",
            f'{FEATURE_SET}.job': "job",
            f'{FEATURE_SET}.duration': "duration",
            f'{FEATURE_SET}.credit_amount': "credit_amount",
            f'{FEATURE_SET}.housing': "housing",
            f'{FEATURE_SET}.purpose': "purpose",
            f'{FEATURE_SET}.saving_account': "saving_account",
            f'{FEATURE_SET}.sex': "sex",
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