import pandas as pd
import qwak
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, ModelSchema
from transformers import BertTokenizer, BertForMaskedLM
import torch

# Define the BERTTextGeneration class, inheriting from QwakModel
class BERTTextGeneration(QwakModel):

    # Initialize the class with the BERT model ID
    def __init__(self):
        self.bert_model_id = 'bert-base-uncased'  # Pre-trained BERT model ID

    # Build method for logging metrics (placeholder)
    def build(self):
        qwak.log_metric({"val_accuracy": 1})

    # Define the schema for the model's API
    def schema(self):
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),  # Input feature is a string named "prompt"
            ])
        return model_schema


    # Initialize the BERT model and tokenizer
    def initialize_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_id)
        self.model = BertForMaskedLM.from_pretrained(self.bert_model_id)


    # Define the prediction method for text generation
    @qwak.api()
    def predict(self, df):
        # Extract the 'prompt' values from the DataFrame
        input_text = list(df['prompt'].values)[0]  # Assuming a single prompt for simplicity
        
        # Tokenize the input text
        encoded_input = self.tokenizer.encode(input_text, return_tensors='pt')
        
        # Perform text generation without gradient calculation
        with torch.no_grad():
            output = self.model.generate(encoded_input, max_length=50)  # Max length for generated text
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Create a DataFrame with the generated text
        return pd.DataFrame([
            {
                "generated_text": generated_text
            }
        ])
