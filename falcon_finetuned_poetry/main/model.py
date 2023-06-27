import pandas as pd
import qwak
from pandas import DataFrame
from qwak.model.base import QwakModel
from qwak.model.schema import ModelSchema, ExplicitFeature
from qwak.model.tools import run_local

from falcon_training import train_model
from generate import init_pretrained_model
from helpers import get_device


class FalconFinetunedModel(QwakModel):

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = get_device()
        # self.model_id = "tiiuae/falcon-7b-instruct"
        self.model_id = "ybelkada/falcon-7b-sharded-bf16"

    def build(self):
        self.model, self.tokenizer = train_model(self.model_id)

    def schema(self):
        return ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),
            ])

    def initialize_model(self):
        self.model, self.tokenizer = init_pretrained_model(self.model_id)

    @qwak.api()
    def predict(self, df):
        prompt = list(df['prompt'].values)[0]
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"],
            max_new_tokens=10,
            pad_token_id=self.tokenizer.eos_token_id
        )
        decoded_output = self.tokenizer.decode(outputs[0],
                                               skip_special_tokens=True)
        return pd.DataFrame(decoded_output)


if __name__ == '__main__':
    m = FalconFinetunedModel()
    input_ = DataFrame(
        [{
            "prompt": "Why does it matter if a Central Bank has a negative rather than 0% interest rate?",
        }]
    ).to_json()
    run_local(m, input_)
