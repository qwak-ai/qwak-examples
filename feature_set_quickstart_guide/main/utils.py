from typing import Tuple
import pandas as pd

# Utility function
def features_cleaning(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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