import qwak
from qwak.model.tools import run_local
from qwak.model.base import QwakModel
from qwak.model.schema import ModelSchema, ExplicitFeature
from sentence_transformers import SentenceTransformer
from pandas import DataFrame
from helpers import get_device


class SentenceEmbeddingsModel(QwakModel):

    def __init__(self):
        self.model_id = "sentence-transformers/all-MiniLM-L12-v2"
        self.model = None
        self.device = None

    def build(self):
        qwak.log_metric({"val_accuracy": 1})

    def schema(self):
        return ModelSchema(
            inputs=[
                ExplicitFeature(name="text", type=str),
            ])

    def initialize_model(self):
        self.device = get_device()
        print(f"Inference using device: {self.device}")
        self.model = SentenceTransformer(
            model_name_or_path=self.model_id,
            device=self.device,
        )

    @qwak.api()
    def predict(self, df):
        text_embeds = self.model.encode(
            df['input'].values.tolist(),
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=self.device,
            batch_size=128,
        ).tolist()
        return DataFrame({"embeddings": text_embeds})


if __name__ == '__main__':
    m = SentenceEmbeddingsModel()
    input_ = DataFrame(
        [{
            "input": "Why does it matter if a Central Bank has a negative rather than 0% interest rate?"
        }]
    ).to_json()
    print(run_local(m, input_))
