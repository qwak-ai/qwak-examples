import qwak
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, ModelSchema
from transformers import pipeline


class GPTNeoModel(QwakModel):

    def __init__(self):
        self.model_id = 'EleutherAI/gpt-neo-125M'
        self.model = None

    def build(self):
        pass

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
        return self.model(
            list(df['prompt'].values),
            do_sample=True,
            max_new_tokens=100
        )
