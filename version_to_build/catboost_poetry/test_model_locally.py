from pandas import DataFrame
from qwak.model.tools import run_local
from main import *
from catboost import CatBoostClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def predict_catboost_model():
    # Load a sample dataset (Iris dataset)
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_test
    # # Make predictions on the test set
    # y_pred = model.predict(X_test)
    #
    # # Calculate accuracy
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy * 100: .2f}%")
    #
    # print("Feature Importances: ", model.get_feature_importance())


if __name__ == '__main__':
    # Create a new instance of the model
    m = load_model()

    feature_vector = [
        {
            "UserId": "male",
            "Age": 3,
            "Sex": "male",
            "Job": 2,
            "Housing": "male",
            "Saving accounts": "male",
            "Checking account": "male",
            "Credit amount": 54.2,
            "Duration": 4,
            "Purpose": "male",
            "Age_cat": "male",

        }]

    df_vector = predict_catboost_model()
    df = DataFrame(df_vector).to_json()

    # Create the DataFrame and convert it to JSON
    # df = DataFrame(feature_vector).to_json()
    print("Predicting for: \n\n", df)

    # Run local inference using the model and print the prediction
    # The run_local function is part of the qwak library and allows for local testing of the model
    prediction = run_local(m, df)
    print("\nPrediction: ", prediction)