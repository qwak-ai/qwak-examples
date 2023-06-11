import os

import pandas as pd
import qwak
from pandas import DataFrame
from qwak.model.base import QwakModel
from qwak.model.schema import ModelSchema, ExplicitFeature
from qwak.model.tools import run_local
from transformers import T5Tokenizer

from helpers import train_model
from utils import load_data

RUNNING_FILE_ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


class FineTuneFLANT5Model(QwakModel):

    def __init__(self):
        self.model_id = "t5-small"
        self.max_new_tokens = 100
        self.model = None
        self.tokenizer = None
        self.model_params = {
            "model": self.model_id,
            "train_batch_size": 8,
            "valid_batch_size": 8,
            "train_epochs": 3,
            "val_epochs": 1,
            "learning_rate": 1e-4,
            "max_source_text_length": 512,
            "max_target_text_length": 50,
            "seed": 42,
            "data_rows": 1000,
            "input_path": f"{RUNNING_FILE_ABSOLUTE_PATH}/data.csv"
        }

    def build(self):
        """
        Training the T5 model
        """
        dataframe = load_data(
            max_length=self.model_params["data_rows"],
            input_path=self.model_params["input_path"]
        )
        # Adding the summarization request to each training row
        dataframe["text"] = "summarize: " + dataframe["text"]

        self.model = train_model(
            dataframe=dataframe,
            source_text="text",
            target_text="headlines",
            output_dir="outputs",
            model_params=self.model_params,
        )

    def schema(self):
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),
            ])
        return model_schema

    def initialize_model(self):
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_id,
            model_max_length=self.model_params["max_source_text_length"]
        )

    @qwak.api()
    def predict(self, df):
        """
        Extracting text the dataframe and encoding it for the model
        :param df:
        :return:
        """
        # Tokenizing input text
        input_text = list(df['prompt'].values)
        input_ids = self.tokenizer(
            input_text,
            return_tensors="pt"
        )
        # Generating the model prediction
        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=self.max_new_tokens
        )
        # Decoding the model output into text
        decoded_outputs = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        return pd.DataFrame([
            {
                "generated_text": decoded_outputs
            }
        ])


if __name__ == '__main__':

    model = FineTuneFLANT5Model()
    vector = {
        'prompt': """Investigators searching for a lost plane carrying Argentine forward Emiliano Sala found two seat cushions on French coast that "likely" belonged to the aircraft. The investigators said they'll now launch an underwater seabed search for aircraft wreckage. The Cardiff City footballer was travelling from France's Nantes to Wales' Cardiff when his plane disappeared over English Channel on January 21."""
    }
    input_ = DataFrame([vector]).to_json()

    res = run_local(model, input_)
    print(res)
