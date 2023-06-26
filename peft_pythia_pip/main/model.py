from pathlib import Path

import pandas as pd
import qwak
import torch
from pandas import DataFrame
from qwak.model.base import QwakModel
from qwak.model.schema import ModelSchema, ExplicitFeature
from qwak.model.tools import run_local

from lit.finetune.adapter import setup
from lit.generate.adapter import load_model, generate_prediction
from lit.prepare_dataset import prepare
from lit.scripts.convert_hf_checkpoint import convert_hf_checkpoint
from lit.scripts.download import download_from_hub
from lit.scripts.prepare_alpaca import generate_prompt


# self.model_id = 'togethercomputer/RedPajama-INCITE-Base-3B-v1'
# self.model_id = "EleutherAI/pythia-160m"


class PEFTModel(QwakModel):

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.fabric = None
        self.model_id = "EleutherAI/pythia-70m"
        self.checkpoint_path = Path("checkpoints/" + self.model_id)
        self.model_params = {
            "max_new_tokens": 100,
            "top_k": 200,
            "temperature": 0.8,
        }
        torch.set_float32_matmul_precision('high')

    def build(self):
        # Get the initial model weights
        download_from_hub(self.model_id)

        # Prepare the weights
        convert_hf_checkpoint(checkpoint_dir=self.checkpoint_path)

        # Prepare the dataset for fine-tuning for training and validation
        prepare(checkpoint_dir=self.checkpoint_path)

        # Fine-tune the model
        setup(checkpoint_dir=self.checkpoint_path)

        # Logging metrics for automation
        qwak.log_metric({"val_accuracy": 1})

    def schema(self):
        return ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),
            ])

    def initialize_model(self):
        self.model, self.tokenizer, self.fabric = load_model(
            checkpoint_dir=self.checkpoint_path
        )

    @qwak.api()
    def predict(self, df):
        input_prompt = list(df['prompt'].values)[0]
        sample = {"instruction": input_prompt, "input": input}
        prompt = generate_prompt(sample)

        encoded = self.tokenizer.encode(prompt, device=self.model.device)

        output = generate_prediction(self.model,
                                     self.tokenizer,
                                     self.fabric,
                                     encoded=encoded,
                                     max_new_tokens=self.model_params["max_new_tokens"],
                                     temperature=self.model_params["temperature"],
                                     top_k=self.model_params["top_k"]
                                     )

        return pd.DataFrame([{
            "generated_text": output
        }])


if __name__ == '__main__':
    m = PEFTModel()
    input_ = DataFrame(
        [{
            "prompt": "Why does it matter if a Central Bank has a negative rather than 0% interest rate?"
        }]
    ).to_json()
    run_local(m, input_)
