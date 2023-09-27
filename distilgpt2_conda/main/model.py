import pandas as pd
import qwak
import torch
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, ModelSchema
from transformers import AutoModelForCausalLM, AutoTokenizer


# Define the DistilGPT2Model class, inheriting from QwakModel
class DistilGPT2Model(QwakModel):


    # Initialize the model attributes
    def __init__(self):
        self.model_id = 'distilgpt2'  # Pre-trained model ID
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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )


    # Define the predict method
    @qwak.api()
    def predict(self, df):
        # Extract 'prompt' values from the DataFrame
        prompts = list(df['prompt'].values)

        # Set padding token if not already set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set padding side
        self.tokenizer.padding_side = "left"

        # Set device and model evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device=device, dtype=torch.bfloat16)

        # Define parameters for text generation
        generate_kwargs = {
            "temperature": 0.5,
            "top_p": 0.92,
            "top_k": 0,
            "max_new_tokens": 512,
            "use_cache": True,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "repetition_penalty": 1.1,
        }

        # Tokenize the input text and move to model's device
        input_ids = self.tokenizer(prompts, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)

        # Generate text based on the input
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **generate_kwargs)

        # Extract new tokens from the generated output
        new_tokens = output_ids[0, len(input_ids[0]):]
        decoded_outputs = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Return the generated text as a DataFrame
        return pd.DataFrame([
            {
                "generated_text": decoded_outputs
            }
        ])