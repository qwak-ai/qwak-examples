# Import required libraries
import pandas as pd
import qwak
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, ModelSchema
from transformers import AutoModelForCausalLM, AutoTokenizer


# Define the CodeGenModel class, inheriting from QwakModel
class CodeGenModel(QwakModel):


    # Initialize the model attributes
    def __init__(self):
        self.model_id = 'Salesforce/codegen-350M-mono'  # Pre-trained model ID
        self.max_new_tokens = 100  # Maximum number of new tokens to generate
        self.model = None  # Placeholder for the model
        self.tokenizer = None  # Placeholder for the tokenizer



    # Build the model (currently just logs a metric)
    def build(self):
        qwak.log_metric({"val_accuracy": 1})  # Log validation accuracy as 1 (placeholder)



    # Define the schema for the model's input
    def schema(self):
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),  # Input feature "prompt" of type string
            ])
        return model_schema



    # Initialize the pre-trained model and its tokenizer
    def initialize_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)  # Load pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)  # Load tokenizer



    # Define the predict method
    @qwak.api()
    def predict(self, df):
        input_text = list(df['prompt'].values)  # Extract 'prompt' values from the DataFrame
        input_ids = self.tokenizer(input_text, return_tensors="pt")  # Tokenize the input text

        # Define parameters for text generation
        params = {
            "pad_token_id": self.tokenizer.eos_token_id,  # End-of-sentence token ID
            "max_new_tokens": self.max_new_tokens  # Maximum number of new tokens
        }

        # Generate text based on the input
        outputs = self.model.generate(**input_ids, **params)
        decoded_outputs = self.tokenizer.decode(outputs[0])  # Decode the generated text

        # Return the generated text as a DataFrame
        return pd.DataFrame([
            {
                "generated_text": decoded_outputs
            }
        ])
