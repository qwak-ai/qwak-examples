import os

import qwak
import torch
from qwak.model.adapters import JsonOutputAdapter
from qwak.model.base import QwakModel
from qwak.model.schema import ModelSchema, ExplicitFeature
from sentence_transformers import SentenceTransformer

from helpers import get_device


class SentenceEmbeddingsModel(QwakModel):

    def __init__(self):
        self.model_id = "sentence-transformers/all-MiniLM-L12-v2"
        self.model = None
        self.device = None
        self.pid = None

    def build(self):
        pass

    def schema(self):
        return ModelSchema(
            inputs=[
                ExplicitFeature(name="text", type=str),
            ])

    def initialize_model(self):
        self.pid = os.getpid()
        self.device = get_device(self.pid)
        print(f"PID is {self.pid}, device count is {torch.cuda.device_count()}")
        print(f"Using device type: {self.device.type} with index: {self.device.index}")
        self.model = SentenceTransformer(
            model_name_or_path=" ",
            device=self.device,
        )

    @qwak.api(output_adapter=JsonOutputAdapter())
    def predict(self, df):
        data = list(df['text'].values)
        text_embeds = self.model.encode(
            data,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=self.device,
            batch_size=128,
        ).tolist()
        return text_embeds
