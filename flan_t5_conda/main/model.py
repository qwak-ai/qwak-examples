from qwak.model.tools import run_local
import pandas as pd
import qwak
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, ModelSchema
from transformers import T5Tokenizer, T5ForConditionalGeneration


class FLANT5BaseModel(QwakModel):

    def __init__(self):
        self.model_id = "google/flan-t5-base"
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
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_id)

    @qwak.api()
    def predict(self, df):
        input_text = list(df['prompt'].values)
        input_ids = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**input_ids,
                                      max_new_tokens=self.max_new_tokens)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return decoded_outputs


if __name__ == '__main__':
    from qwak.model.tools import run_local
    import pandas as pd

    model = FLANT5BaseModel()
    input_ = [
        {"prompt": "what is love?"}
    ]
    response = run_local(model, pd.DataFrame(input_).to_json())
    print(response)

