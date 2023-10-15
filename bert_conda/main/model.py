import pandas as pd
import qwak
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, ModelSchema
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Define the BERTSentimentAnalysis class, inheriting from QwakModel
class BERTSentimentAnalysis(QwakModel):

    # Initialize the class with the BERT pretrained
    def __init__(self):
        self.bert_model_id = 'textattack/bert-base-uncased-SST-2'


    # Logging random metric, this model will use pre-trained Bert
    def build(self):
        qwak.log_metric({"val_accuracy": 1})


    # Define the schema for the QwakModel's API
    def schema(self):
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),  # Input feature is a string named "prompt"
            ])
        return model_schema


    # Initialize the BERT model and tokenizer from a pretrained
    def initialize_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_id)
        self.model = BertForSequenceClassification.from_pretrained(self.bert_model_id)


    # Predict method will analyze sentiment for given request input
    @qwak.api()
    def predict(self, df):
        # Extract the 'prompt' values from the DataFrame
        input_text = list(df['prompt'].values)
        
        # Tokenize the input text
        encoded_input = self.tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
        
        # Perform inference without gradient calculation
        with torch.no_grad():
            output_tuple = self.model(**encoded_input)
        
        # Extract logits and compute predictions
        logits = output_tuple.logits
        predictions = torch.argmax(logits, dim=-1)  # 0 is Negative, 1 is Positive

        # Create a DataFrame with the sentiment results
        return pd.DataFrame([
            {
                "sentiment": "Positive" if pred.item() == 1 else "Negative"
            } for pred in predictions
        ])
