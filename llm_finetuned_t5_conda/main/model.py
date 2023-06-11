import pandas as pd
from pandas import DataFrame
from qwak.model.tools import run_local
import qwak
from qwak.model.base import QwakModel
from transformers import T5Tokenizer

from utils import load_data
from trainer import T5Trainer


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
            "MODEL": "t5-small",  # model_type: t5-base/t5-large
            "TRAIN_BATCH_SIZE": 8,  # training batch size
            "VALID_BATCH_SIZE": 8,  # validation batch size
            "TRAIN_EPOCHS": 1,  # number of training epochs
            "VAL_EPOCHS": 1,  # number of validation epochs
            "LEARNING_RATE": 1e-4,  # learning rate
            "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
            "MAX_TARGET_TEXT_LENGTH": 50,  # max length of target text
            "SEED": 42,  # set seed for reproducibility
        }
        self.input_path = "https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv"

    def build(self):
        dataframe = load_data(
            max_length=10,
            input_path=self.input_path
        )
        dataframe["text"] = "summarize: " + dataframe["text"]

        self.model = T5Trainer(
            dataframe=dataframe,
            source_text="text",
            target_text="headlines",
            model_params=self.model_params,
            output_dir="outputs",
        )

    def schema(self):
        pass

    def initialize_model(self):
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_id)

    @qwak.api()
    def predict(self, df):
        input_text = list(df['prompt'].values)
        input_ids = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**input_ids,
                                      max_new_tokens=self.max_new_tokens)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return pd.DataFrame([
            {
                "generated_text": decoded_outputs
            }
        ])


if __name__ == '__main__':
    model = FineTuneFLANT5Model()
    vector = {
        'prompt': "what is love?"
    }
    input_ = DataFrame([vector]).to_json()

    res = run_local(model, input_)
    print(res)
