from qwak.model.tools import run_local

import torch
import pandas as pd
import qwak
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, ModelSchema
from transformers import AutoModelForCausalLM, AutoTokenizer


class DistilGPT2Model(QwakModel):

    def __init__(self):
        self.model_id = 'distilgpt2'
        self.model = None
        self.tokenizer = None

    def build(self):
        pass

    def schema(self):
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),
            ])
        return model_schema

    def initialize_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        qwak.log_metric({"val_accuracy": 1})

    @qwak.api()
    def predict(self, df):
        # Taking the prompt values from the input dataframe
        prompts = list(df['prompt'].values)

        # "pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id."
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device=device, dtype=torch.bfloat16)

        generate_kwargs = {
            "temperature": 0.5,
            "top_p": 0.92,
            "top_k": 0,
            "max_new_tokens": 512,
            "use_cache": True,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": 1.1,  # 1.0 means no penalty, > 1.0 means penalty, 1.2 from CTRL paper
        }

        input_ids = self.tokenizer(prompts, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)
        gkw = {**generate_kwargs}

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **gkw)

        # Slice the output_ids tensor to get only new tokens
        new_tokens = output_ids[0, len(input_ids[0]):]
        decoded_outputs = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return pd.DataFrame([
            {
                "generated_text": decoded_outputs
            }
        ])
