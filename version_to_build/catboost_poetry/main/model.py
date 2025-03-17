import frogml
import pandas as pd
import qwak
from qwak.model.base import QwakModel


class CatBoostModel(QwakModel):

    def __init__(self):

        self.params = {
            "repository": "frogml-tests",
            "model_name": "catboost-sample",
            "version": "1.0.0"
        }
        self.model = None

    def build(self):
        """
        The build() method is called once during the build process.
        Use it for training or actions during the model build phase.
        """
        pass

    def schema(self):
        """
        schema() define the model input structure, and is used to enforce 
        the correct structure of incoming prediction requests.
        """
        pass

    def initialize_model(self):
        self.model = frogml.catboost.load_model(
            repository=self.params["repository"],
            model_name=self.params["model_name"],
            version=self.params["version"],
        )

    @qwak.api()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The predict(df) method is the live inference method 
        """
        return pd.DataFrame(
            self.model.predict(df)
        )
