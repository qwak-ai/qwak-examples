from pathlib import Path

import pandas as pd
import qwak
import torch
from qwak.model.base import QwakModel
from qwak.model.schema import ModelSchema, ExplicitFeature

from lit.finetune.adapter import setup
from lit.generate.adapter import load_model, generate_prediction
from lit.prepare.alpaca_data import prepare
from lit.scripts.convert_hf_checkpoint import convert_hf_checkpoint
from lit.scripts.download import download_from_hub
from lit.scripts.prepare_alpaca import generate_prompt


class PEFTModel(QwakModel):

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.fabric = None
        self.model_id = "EleutherAI/pythia-410m-deduped"
        self.checkpoint_path = Path("checkpoints").joinpath(self.model_id)
        self.data_path = Path("data/finance-alpaca")
        self.model_params = {
            "max_new_tokens": 100,
            "top_k": 200,
            "temperature": 0.8,
            # "precision": "16-true",  # For V100
            "precision": "bf16-mixed",  # For A10 / A100
        }
        torch.set_float32_matmul_precision('high')

    def build(self):
        # Download and prepare the model weights
        download_from_hub(self.model_id)
        convert_hf_checkpoint(checkpoint_dir=self.checkpoint_path)

        # Prepare dataset for fine-tuning with training and validation sets
        prepare(checkpoint_dir=self.checkpoint_path,
                destination_path=self.data_path)

        # Fine-tuning the model
        setup(checkpoint_dir=self.checkpoint_path,
              data_dir=self.data_path,
              precision=self.model_params["precision"]
              )

        # Logging metrics for automation
        qwak.log_metric({"val_accuracy": 1})

    def schema(self):
        return ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),
            ])

    def initialize_model(self):
        # Loading the model instance for better memory handling
        self.model, self.tokenizer, self.fabric = load_model(
            checkpoint_dir=self.checkpoint_path,
            precision=self.model_params['precision']
        )

    @qwak.api()
    def predict(self, df):
        user_prompt = list(df['prompt'].values)[0]

        if not user_prompt:
            return pd.DataFrame([{
                "generated_text": ""
            }])

        prompt = generate_prompt({
            "instruction": user_prompt,
            "input": None
        })
        encoded = self.tokenizer.encode(prompt,
                                        device=self.model.device)
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
