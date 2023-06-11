import pandas as pd
import qwak
from pandas import DataFrame
from qwak.model.base import QwakModel
from qwak.model.schema import ModelSchema, ExplicitFeature
from qwak.model.tools import run_local
from transformers import T5Tokenizer

from trainer import T5Trainer
from utils import load_data


class FineTuneFLANT5Model(QwakModel):

    def __init__(self):
        self.model_id = "google/flan-t5-small"
        self.max_new_tokens = 100
        self.model = None
        self.tokenizer = None
        self.max_new_tokens = 100
        self.model = None
        self.trainer = None
        self.model_params = model_params = {
            "MODEL": "t5-small",
            "TRAIN_BATCH_SIZE": 8,
            "VALID_BATCH_SIZE": 8,
            "TRAIN_EPOCHS": 1,
            "VAL_EPOCHS": 1,
            "LEARNING_RATE": 1e-4,
            "MAX_SOURCE_TEXT_LENGTH": 512,
            "MAX_TARGET_TEXT_LENGTH": 50,
            "SEED": 42,
            "input_path": "https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv"
        }

    def build(self):
        """
        Training the T5 model
        """
        dataframe = load_data(
            max_length=10,
            input_path=self.model_params["input_path"]
        )
        # Adding the summarization request to each training row
        dataframe["text"] = "summarize: " + dataframe["text"]

        self.model = T5Trainer(
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
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_id)

    @qwak.api()
    def predict(self, df):
        """
        Extracting text the dataframe and encoding it for the model
        :param df:
        :return:
        """
        input_text = list(df['prompt'].values)
        # Tokenizing input text
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
        'prompt': "Today I wanted to go to school but it was raining"
    }
    input_ = DataFrame([vector]).to_json()

    res = run_local(model, input_)
    print(res)
