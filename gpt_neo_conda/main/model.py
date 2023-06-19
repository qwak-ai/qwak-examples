import pandas as pd
import qwak
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, ModelSchema
from transformers import pipeline


class GPTNeoModel(QwakModel):

    def __init__(self):
        self.model_id = 'EleutherAI/gpt-neo-125M'
        self.model = None
        self.max_new_tokens = 100

    def build(self):
        qwak.log_metric({"val_accuracy": 1})

    def schema(self):
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),
            ])
        return model_schema

    def initialize_model(self):
        self.model = pipeline('text-generation',
                              model=self.model_id,
                              pad_token_id=50256)

    @qwak.api()
    def predict(self, df):
        decoded_outputs = self.model(
            list(df['prompt'].values),
            do_sample=True,
            max_new_tokens=self.max_new_tokens
        )

        return pd.DataFrame(decoded_outputs)
