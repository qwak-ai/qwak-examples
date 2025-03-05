from typing import Optional

import frogml
import pandas as pd
import torch
from frogml_core.model.adapters import DataFrameOutputAdapter
from pandas import DataFrame
from qwak.model.base import QwakModel
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification


class HuggingFaceModel(QwakModel):

    def __init__(self):
        self.model: Optional[DistilBertModel] = None
        self.tokenizer: Optional[DistilBertTokenizer] = None
        self.device = None
        self.model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

        self.jfrog_repository = "version-to-build"
        self.jfrog_model_name = "huggingface"

    def build(self):
        """
        The build() method is called once during the build process.
        Use it for training or actions during the model build phase.
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.model_name
        )

        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name
        )

        frogml.huggingface.log_model(
            repository=self.jfrog_repository,
            model_name=self.jfrog_model_name,
            model=self.model,
            tokenizer=self.tokenizer,
            parameters={
                "evaluation": False
            },
            metrics={
                "test": "test"
            }
        )

    def schema(self):
        """
        schema() define the model input structure, and is used to enforce
        the correct structure of incoming prediction requests.
        """
        pass

    def initialize_model(self):
        self.model, self.tokenizer = frogml.huggingface.load_model(
            repository=self.jfrog_repository,
            model_name=self.jfrog_model_name,
            version="2025-03-03-13-58-06-651",
        )
        # Check if a GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Setting device as {self.device}")
        self.model.to(self.device)

    @frogml.api()
    def predict(self, df: DataFrame, analytics_logger = None) -> DataFrame:
        inputs = self.tokenizer(
            df["text"].to_list(), return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        num_spaces = df["text"].apply(lambda x: x.count(" "))
        num_words = df["text"].apply(lambda x: len(x.split()))
        length = df["text"].apply(len)

        if analytics_logger:
            analytics_logger.log_multi(
                values={
                    "sentence_length": str(length[0]),
                    "num_spaces": str(num_spaces[0]),
                    "num_words": str(num_words[0]),
                }
            )
        with torch.no_grad():
            logits = self.model(**inputs).logits

        probabilities = logits.softmax(dim=1).cpu().numpy()
        predicted_labels = [
            self.model.config.id2label[class_id.argmax()] for class_id in probabilities
        ]
        results = DataFrame(
            list(zip(predicted_labels, probabilities[:, 1])), columns=["label", "score"]
        )

        return results

if __name__ == "__main__":
    model = HuggingFaceModel()
    model.build()
    model.initialize_model()
    input = pd.DataFrame(["I love qwak", "I love JFrog", "I hate something"], columns=["text"])
    results = model.predict(input)