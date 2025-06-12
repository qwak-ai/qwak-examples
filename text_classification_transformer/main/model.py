import pandas as pd
import numpy as np
from frogml import FrogMlModel
from frogml_core.model.schema import ExplicitFeature, ModelSchema
import frogml
import frogml.huggingface
import os

import torch
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,)

import evaluate
from frogml_core.tools.logger import get_frogml_logger

logger = get_frogml_logger()

JF_REPOSITORY = 'nlp-models'
JF_MODEL = os.environ['QWAK_MODEL_ID']
JF_VERSION = os.environ['MODEL_VERSION']
HF_MODEL = 'distilbert-base-uncased'

class TextClassification(FrogMlModel):

    # ----- Class Initialization -----
    def __init__(self):
        self.train = os.environ.get('TRAIN', None)

    def _load_and_tokenize_dataset(self): 

        from datasets import load_dataset

        dataset = load_dataset("ag_news")

        small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
        small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(500))

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True)

        train_tokenized_dataset = small_train_dataset.map(preprocess_function, batched=True)
        test_tokenized_dataset = small_eval_dataset.map(preprocess_function, batched=True)

        return train_tokenized_dataset, test_tokenized_dataset


    # ----- Set the PyTorch device based on the hardware detected -----
    def set_torch_device(self):

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("MPS (Metal) is available. Using GPU.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("CUDA is available. Using GPU.")
        else:
            self.device = torch.device("cpu")
            print("CUDA or MPS not available. Using CPU.")

        print('Using device:', self.device)

        #Additional Info when using cuda
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


    # ----- TRAINING LOGIC ------
    def build(self):

        self.set_torch_device()

        if self.train:

            train_dataset, eval_dataset = self._load_and_tokenize_dataset()

            self.model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL, num_labels=4)

            # 4. Use a Data Collator to handle padding
            # This is more efficient as it pads batches to the length of the longest item
            # in that batch, not to the overall maximum length.
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

            metric = evaluate.load("accuracy")

            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                return metric.compute(predictions=predictions, references=labels)


            training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

            # 7. Define Training Arguments
            training_args = TrainingArguments(
                output_dir="./results",
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                num_train_epochs=3,
                weight_decay=0.01,
            )

            # 8. Create the Trainer instance
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator, # This is the key change
                compute_metrics=compute_metrics,
            )

            trainer.train()

            # Evaluate the model
            eval_result = trainer.evaluate()


            # Print the evaluation metrics
            print(f"Evaluation Result: {eval_result}")

            frogml.huggingface.log_model(
                            model = self.model,
                            tokenizer = self.tokenizer,
                            model_name = JF_MODEL,
                            repository = JF_REPOSITORY,
                            version = JF_VERSION,
                            metrics = eval_result
                        )


    # ----- RUNTIME INITIALIZATION LOGIC -----
    def initialize_model(self):

        self.model, self.tokenizer = frogml.huggingface.load_model(
            repository= JF_REPOSITORY,
            model_name= JF_MODEL,
            version= JF_VERSION,
        )

        self.set_torch_device()

        print(f"Setting device as {self.device}")
        self.model.to(self.device)


    # ----- INPUT/OUTPUT SCHEMA ------
    def schema(self):

        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="text", type=str),
            ])
        return model_schema


    # ----- INFERENCE LOGIC ------
    @frogml.api(analytics=True)
    def predict(self, df: pd.DataFrame, analytics_logger = None) -> pd.DataFrame:

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
        results = pd.DataFrame(
            list(zip(predicted_labels, probabilities[:, 1])), columns=["label", "score"]
        )

        return results
