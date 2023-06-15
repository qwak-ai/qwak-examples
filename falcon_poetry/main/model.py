from qwak import QwakModelInterface
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import pandas as pd
import qwak
from pandas import DataFrame
from qwak.model.schema import ModelSchema, ExplicitFeature


class FalconModel(QwakModelInterface):

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id = "tiiuae/falcon-7b"

    def build(self):
        pass

    def schema(self):
        return ModelSchema(
            features=[
                ExplicitFeature(name="prompt", type=str),
            ])

    def initialize_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # model = AutoModelForCausalLM.from_pretrained(self.model_id, 
        #                                              trust_remote_code=True,
        #                                              torch_dtype=torch.bfloat16,
        #                                              device_map="auto",
        #                                             )

        self.model = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    @qwak.api()
    def predict(self, df):
        decoded_outputs = self.model(
            list(df['prompt'].values),
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return pd.DataFrame(decoded_outputs)


if __name__ == '__main__':
    model = FalconModel()
    input_ = DataFrame([{
        "prompt": "what is love?"
    }])
    model.run_local(input_.to_json())