import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Tuple

class DataPreprocessor:  # Encapsulate the scaler

    def __init__(self):
        self.scaler = StandardScaler()  # Initialize the scaler

    def preprocess_training_data(self, input_df: pd.DataFrame, test_size: float = 0.3, validation_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Preprocesses the input DataFrame for training by splitting it into training, validation,
        and test sets, scaling the numerical features, and calculating class weights.

        Args:
            input_df: The input DataFrame containing the features and the target variable ('Class').
            test_size: The proportion of the data to use for the test set.
            validation_size: The proportion of the training data to use for the validation set.
            random_state: Random seed for reproducibility.

        Returns:
            A tuple containing:
                - X_train: The scaled training features.
                - X_validate: The scaled validation features.
                - X_test: The scaled test features.
                - y_train: The training target variable.
                - y_validate: The validation target variable.
                - y_test: The test target variable.
                - w_p: The weight for non-fraudulent transactions in the training set.
                - w_n: The weight for fraudulent transactions in the training set.
        """
        X = input_df.drop('Class', axis=1)
        y = input_df['Class']

        # First split into training and test sets, stratifying by 'y'
        X_train_v, X_test, y_train_v, y_test = train_test_split(X, y,
                                                            test_size=test_size, random_state=random_state,
                                                            stratify=y)
        # Then split the training set into training and validation sets, again stratifying
        X_train, X_validate, y_train, y_validate = train_test_split(X_train_v, y_train_v,
                                                                    test_size=validation_size, random_state=random_state,
                                                                    stratify=y_train_v)

        X_train = self.scaler.fit_transform(X_train)  # Fit on training data
        X_validate = self.scaler.transform(X_validate)
        X_test = self.scaler.transform(X_test)

        w_p = y_train.value_counts()[0] / len(y_train)
        w_n = y_train.value_counts()[1] / len(y_train)
        print(f"Fraudulent transaction weight: {w_n}")
        print(f"Non-Fraudulent transaction weight: {w_p}")

        return (X_train, X_test, y_train, y_test)

    def preprocess_inference_data(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        Preprocesses the input DataFrame for inference by scaling the numerical features
        using the fitted StandardScaler.

        Args:
            input_df: The input DataFrame containing the features.

        Returns:
            The scaled features as a NumPy array.
        """
        X = input_df.drop('Class', axis=1, errors='ignore')  # Handle missing 'Class'
        X_scaled = self.scaler.transform(X)  # Use the fitted scaler
        return X_scaled