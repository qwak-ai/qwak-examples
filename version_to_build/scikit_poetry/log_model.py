import frogml
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from qwak.model.base import QwakModel


class IrisClassifier(QwakModel):

    def __init__(self):
        pass

    def build(self):
        """Train a RandomForest model on the Iris dataset."""
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        frogml.scikit_learn.log_model(
            model=self.model,
            model_name="scikit",
            repository="version-to-build",
            parameters={
                "training": False
            }
        )

        print("Model trained with accuracy:", self.model.score(X_test, y_test))

    @frogml.api(analytics=True)
    def predict(self, data):
        """Run inference on input data."""
        return self.model.predict(data).tolist()