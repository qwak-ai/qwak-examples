# Import required libraries
import pandas as pd
import qwak
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, ModelSchema
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

from qwak.model.adapters.output_adapters.qwak_with_default_fallback import AutodetectOutputAdapter

# Define the FLANT5Model class inheriting from QwakModel
class FLANT5Model(QwakModel):


    # Initialize model parameters
    def __init__(self):
        self.model_id = "google/flan-t5-small"
        self.max_new_tokens = 50
        self.model = None
        self.tokenizer = None


    # Log model metrics (for demonstration)
    def build(self):
        qwak.log_metric({"val_accuracy": 1})


    # Define the input schema for the model
    def schema(self):
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),
            ])
        return model_schema


    # Load the pre-trained FLAN-T5 model
    def initialize_model(self):
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_id, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_id)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Generate text based on the input prompt
    @qwak.api(output_adapter=AutodetectOutputAdapter())
    def predict(self, df):
        input_text = list(df['prompt'].values)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        # Generate text
        with torch.no_grad():
            gen_params = {
                "max_length": 50,
                "top_k": 50
            }
            output_ids = self.model.generate(input_ids, **gen_params)

        # Decode the generated text
        decoded_outputs = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return pd.DataFrame([
            {
                "generated_text": decoded_outputs
            }
        ])
