import pandas as pd
import qwak
import torch
import transformers
from qwak.model.base import QwakModel
from qwak.model.schema import ModelSchema, ExplicitFeature
from transformers import AutoTokenizer, AutoModelForCausalLM


class FalconModel(QwakModel):

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id = "tiiuae/falcon-7b-instruct"
#        self.offload_folder = tempfile.mkdtemp()
#        print (self.offload_folder)


    def build(self):
        pass

    def schema(self):
        return ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),
            ])

    def initialize_model(self):


        print("Initializing from a pre-trained falcon-b7 model\n")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

        print("Initialized from a pre-trained falcon-b7 model\n")
        """
        self.model = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            #trust_remote_code=True,
            device_map="auto",
            
            #offload_folder=self.offload_folder
        )
        """

    @qwak.api()
    def predict(self, df):
        """
        print(type(df['prompt'].values))
        inputs = self.tokenizer(df['prompt'].values, return_tensors="pt")
        output = self.model.generate(**inputs, max_length=50, num_return_sequences=1)
        """
        
        prompts = df['prompt'].values.tolist()  # Convert to Python list
        decoded_outputs = []

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=200, do_sample=True, top_k=10, num_return_sequences=1)
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            decoded_outputs.append(decoded_output)
        
        
        
        """

        decoded_outputs = self.model(
            list(df['prompt'].values),
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        """

        return pd.DataFrame(self.tokenizer.decode(output[0], skip_special_tokens=True))
