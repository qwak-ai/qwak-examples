from typing import Optional

import frogml
from qwak.model.base import QwakModel


class ScikitModel(QwakModel):

    def __init__(self):
        self.model: Optional = None
        self.device = None

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
        self.model = frogml.scikit_learn.load_model(
            repository="version-to-build",
            model_name="scikit",
            version="2025-03-05-12-41-51-939",
        )

    @frogml.api(analytics=True)
    def predict(self, df):
        """Run inference on input data."""
        return self.model.predict(df).tolist()
