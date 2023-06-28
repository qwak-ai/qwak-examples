import pandas as pd
import qwak
from pandas import DataFrame
from qwak.model.base import QwakModel
from qwak.model.schema import ModelSchema, ExplicitFeature
from qwak.model.tools import run_local
from transformers import T5Tokenizer

from helpers import load_data, get_device
from training import train_model


class FineTuneFLANT5Model(QwakModel):

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_params = {
            "model_id": "t5-large",
            "train_batch_size": 8,
            "valid_batch_size": 8,
            "train_epochs": 3,
            "val_epochs": 1,
            "learning_rate": 1e-4,
            "max_source_text_length": 512,
            "max_target_text_length": 50,
            "seed": 42,
            "max_rows": 10000,
            "input_path": "https://qwak-public.s3.amazonaws.com/example_data/financial_qa.csv",
            "source_column_name": "instruction",
            "target_column_name": "output"
        }

    def build(self):
        dataframe = load_data(
            input_path=self.model_params["input_path"],
            max_length=self.model_params["max_rows"]
        )
        source = self.model_params["source_column_name"]
        target = self.model_params["target_column_name"]

        qwak.log_metric({"val_accuracy": 1})
        # Adding the summarization request to each training row
        dataframe[source] = "question: " + dataframe[source]
        dataframe[target] = "answer: " + dataframe[target]
        self.model = train_model(
            dataframe=dataframe,
            source_text=source,
            target_text=target,
            output_dir="outputs",
            model_params=self.model_params,
        )

    def schema(self):
        return ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),
            ])

    def initialize_model(self):
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_params["model_id"],
            model_max_length=self.model_params["max_source_text_length"]
        )
        self.device = get_device()
        print(f"Inference using device: {self.device}")

    @qwak.api()
    def predict(self, df):
        # Tokenize input text
        input_ids = self.tokenizer(list(df['prompt'].values), return_tensors="pt").to(self.device)
        # Generate prediction
        outputs = self.model.generate(**input_ids, max_new_tokens=self.model_params["max_target_text_length"])
        # Decode model output
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return pd.DataFrame([{
            "generated_text": decoded_outputs
        }])


if __name__ == '__main__':
    m = FineTuneFLANT5Model()
    input_ = DataFrame(
        [{
            "prompt": "Why does it matter if a Central Bank has a negative rather than 0% interest rate?"
        }]
    ).to_json()
    run_local(m, input_)
