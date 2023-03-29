import os
import pandas as pd
import numpy as np
import qwak
import evaluate
from qwak.model.base import QwakModelInterface
from datasets import load_dataset,load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

RUNNING_FILE_ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "distilbert-base-uncased"        

class HuggingFaceTokenizerModel(QwakModelInterface):

    def __init__(self):
        """
        Initializes model parameters and creates a HuggingFace Tokenizer classifier.
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)

    def build(self):
        """
        The build() method is called once during the remote build process on Qwak.
        We use this method to train a Tokenizer model on a Yelp dataset to detect text polarity
        """

        def tokenize(examples):
            return self.tokenizer(examples['text'], padding='max_length', truncation=True)

        # Load the Yelp Polarity database from HuggingFace for training
        dataset = load_dataset('yelp_polarity')

        print('Tokenizing dataset...')
        tokenized_dataset = dataset.map(tokenize, batched=True)

        print('Splitting data to training and evaluation sets')
        train_dataset = tokenized_dataset['train'].shuffle(seed=42).select(range(50))
        eval_dataset = tokenized_dataset['test'].shuffle(seed=42).select(range(50))

        # We don't need the tokenized dataset so we can let the garbage collector free the memory.
        del tokenized_dataset
        del dataset

        # Defining parameters for the training process
        metric = evaluate.load('accuracy')
        # metric = load_metric('accuracy')


        # A helper method to evaluate the model during training
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=1)
            return metric.compute(predictions=predictions, references=labels)

        training_args = TrainingArguments(
            output_dir='training_output',
            evaluation_strategy='epoch',
            num_train_epochs=1
        )

        # Defining all the training parameters for our tokenizer model
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics
        )

        print('Training the model')
        trainer.train()
    
    @qwak.api()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The predict() method takes a pandas DataFrame object (df) as input and returns 
        another pandas DataFrame object with the prediction output.
        """

        input_data = list(df['text'].values)
        
        # Tokenize the input data using a pre-trained tokenizer
        tokenized = self.tokenizer(input_data, padding='max_length', truncation=True, return_tensors='pt')
        
        # Pass the tokenized data to our trained model
        response = self.model(**tokenized)

        return pd.DataFrame(
            response.logits.softmax(dim=1).tolist()
        )

