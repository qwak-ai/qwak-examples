import qwak
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, ModelSchema
from transformers import BertTokenizer, BertModel
import pandas as pd


class BERTModel(QwakModel):

    def __init__(self):
        self.model_id = 'bert-base-uncased'
        self.max_new_tokens = 100
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
        self.model = BertModel.from_pretrained(self.model_id)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_id)

    @qwak.api()
    def predict(self, df):
        input_text = list(df['prompt'].values)

        encoded_input = self.tokenizer(input_text[0], return_tensors='pt')
        output = self.model(**encoded_input)
        decoded_outputs = self.tokenizer.convert_ids_to_tokens(output)

        return pd.DataFrame([
            {
                "generated_text": decoded_outputs
            }
        ])