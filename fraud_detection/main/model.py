import pandas as pd
import frogml
from frogml import FrogMlModel
from frogml.sdk.model.schema import ExplicitFeature, ModelSchema, InferenceOutput
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from catboost import CatBoostClassifier
from main.data_processor import DataPreprocessor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class FraudDetectionModel(FrogMlModel):

    def __init__(self):

        # Create a Catboost Classifier
        self.model = None

        self.data_preprocessor = DataPreprocessor()

        # Define the parameter grid
        self.param_grid = {
            'iterations': [100, 200, 300, 400],  # Number of boosting iterations
            'learning_rate': [0.01, 0.03, 0.05, 0.1],  # Step size shrinkage
            'depth': [4, 6, 8, 10],  # Depth of trees
            'l2_leaf_reg': [1, 3, 5, 7],  # L2 regularization coefficient
            'border_count': [32, 64, 128],  # Number of splits for numerical features
            'random_strength': [0.1, 0.5, 1], # Amount of randomness to add to the score when choosing the tree structure.
            'verbose': [0] # Suppress verbose output during training
        }

        self.input_dataset = 'main/small_fraud_dataset.csv'


    # ----- TRAINING LOGIC ------
    def build(self):

        # Read Data
        df = pd.read_csv(self.input_dataset)

        # Preprocess data
        X_train, X_test, y_train, y_test = self.data_preprocessor.preprocess_training_data(df)

        classifier = CatBoostClassifier(random_state=42, eval_metric='F1') # Use F1 for fraud, set random_state
        
        # Use StratifiedKFold for cross-validation to handle imbalanced data
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Adjust n_splits

        # Create RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=classifier,
            param_distributions=self.param_grid,
            n_iter=10,  # Number of random combinations to try
            scoring='f1',  # Use F1-score for evaluation (crucial for fraud)
            cv=skf,  # Use stratified k-fold cross-validation
            verbose=1,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )

        # Fit the RandomizedSearchCV object to the training data
        random_search.fit(X_train, y_train)

        # Print the best hyperparameters
        print("Best hyperparameters:", random_search.best_params_)

        # Get the best model
        self.model = random_search.best_estimator_

        # Evaluate the best model
        y_test_pred = self.model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'f1_score': f1_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred)
        }

        frogml.log_param(random_search.best_params_)
        frogml.log_metric(metrics)
        frogml.log_data(pd.DataFrame(X_train), tag = 'train_data')


    # ----- INFERENCE LOGIC ------
    @frogml.api()
    def predict(self, df):

        prediction_data = self.data_preprocessor.preprocess_inference_data(df)

        predictions = self.model.predict(prediction_data)

        return pd.DataFrame({'Fraud': predictions})


    # ----- INPUT/OUTPUT SCHEMA ------
    def schema(self):
        """
        schema() define the model input structure.
        Use it to enforce the structure of incoming requests.
        """
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="Time", type=float), #Time
                *[ExplicitFeature(name=f"V{i+1}", type=float) for i in range(28)], # List comprehension for V0-V29
                ExplicitFeature(name="Amount", type=float), #Amount
            ],
            outputs=[
                InferenceOutput(name="Fraud", type=int)
            ])
        return model_schema
