import qwak
from qwak.model.schema import ModelSchema, ExplicitFeature
from qwak.model.tools import run_local
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
from pandas import DataFrame
from qwak.model.base import QwakModel
from huggingface_hub import login
from qwak.clients.secret_service import SecretServiceClient


class Llama2MT(QwakModel):
    """The Model class inherit QwakModel base class"""

    def __init__(self):
        self.model_id = "meta-llama/Llama-2-7b-chat-hf"
        self.model = None
        self.tokenizer = None

    def build(self):
        secret_service: SecretServiceClient = SecretServiceClient()
        hf_token = secret_service.get_secret("hugging-face")
        login(token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id)

    def schema(self):
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),
            ])
        return model_schema

    def initialize_model(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        secret_service: SecretServiceClient = SecretServiceClient()
        hf_token = secret_service.get_secret("hugging-face")
        login(token=hf_token)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model.to(device=self.device, dtype=torch.bfloat16)
        print(f"Build using device: {self.device}")

    @qwak.api()
    def predict(self, df):
        input_text = list(df['prompt'].values)
        input_ids = self.tokenizer(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        print(f"Inference using device: {self.device}")
        outputs = self.model.generate(**input_ids, max_new_tokens=100)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return pd.DataFrame([{"generated_text": decoded_outputs}])


if __name__ == '__main__':
    model = Llama2MT()
    input = DataFrame(
        [{
            "prompt": "Answer: Can you help me find something fun to do in the weekend in NY?"
        }]
    ).to_json()
    prediction = run_local(model, input)
    print(prediction)