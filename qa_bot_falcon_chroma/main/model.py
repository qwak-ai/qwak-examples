import pandas as pd
import qwak
import torch
import transformers
import safetensors
from qwak.model.base import QwakModel
from qwak.model.schema import ModelSchema, ExplicitFeature, InferenceOutput
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import chromadb
from sentence_transformers import SentenceTransformer

# Define a class for the FalconLLM model, inheriting from QwakModel
class FalconLLM(QwakModel):

    def __init__(self):
        # Initialize the model, tokenizer, model ID, and embedding model
        self.model = None
        self.tokenizer = None
        self.model_id = "tiiuae/falcon-7b-instruct"
        self.embedding_model = None

    def build(self):
        # This method is intended to build the model, currently empty
        pass

    def schema(self):
        # Define the schema of the model, specifying input and output features
        return ModelSchema(
            inputs=[
                ExplicitFeature(name="question", type=str),
            ],
            outputs=[
                InferenceOutput(name="basic_answer", type=str),
                InferenceOutput(name="context_answer", type=str)
            ])  

    def initialize_model(self):
        # Set up the device for running the model (GPU if available)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Initialize the sentence transformer model for embedding generation
        self.embedding_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1').to(self.device)
        # Initialize the tokenizer from the pretrained model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # Set up the pipeline for text generation with the specified model and tokenizer
        self.pipeline = transformers.pipeline("text-generation",
                                         model=self.model_id,
                                         tokenizer=self.tokenizer,
                                         torch_dtype=torch.bfloat16,
                                         trust_remote_code=True,
                                         device_map="auto")
        
        # Populate the embeddings in the vector database
        self.populate_embeddings()

    # Decorator to expose the predict function as a Qwak API endpoint
    @qwak.api()
    def predict(self, df):
        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            question = row['question']
            
            # Generate a basic answer without context
            basic_answer = self.generate_answer(question)
            # Query the vector database for context
            context = "".join(self.query_vector(question)['context'][0]) 
            # Create a prompt with context if available
            prompt = question if context is None else f"{context}\n\n{question}"
            # Generate an answer with the context
            context_answer = self.generate_answer(prompt)
            
            # Update the DataFrame with the generated answers
            df.at[index, 'basic_answer'] = basic_answer
            df.at[index, 'context_answer'] = context_answer

        return df

    def populate_embeddings(self):
        # Populate the vector database with embeddings from a dataset
        print("Starting to populate embeddings")
        dataset = load_dataset("databricks/databricks-dolly-15k", split='train')
        closed_qa_dataset = dataset.filter(lambda example: example['category'] == 'closed_qa')

        self.vector_client = chromadb.Client()
        self.vector_collection = self.vector_client.create_collection(name="knowledge-base")

        for i in range(len(closed_qa_dataset)):
            self.vector_collection.add(
                    embeddings=[self.embedding_model.encode(f"{closed_qa_dataset[i]['instruction']}. {closed_qa_dataset[i]['context']}").tolist()],
                    documents=[closed_qa_dataset[i]['context']],
                    ids=[f"id_{i}"]
            )

        print("Finished populating embeddings")

    def generate_answer(self, input):
        # Generate an answer from the input using the text generation pipeline
        sequences = self.pipeline(input, 
                        max_length=200,
                        do_sample=True,
                        top_k=10,
                        num_return_sequences=1,
                        eos_token_id=self.tokenizer.eos_token_id)
        
        print(f"generate_answer context: {sequences}\n")

        return sequences['generated_text']
        

    def query_vector(self, natural_input):
        # Query the vector database for context based on the input
        context = self.vector_collection.query(query_embeddings=[self.embedding_model.encode(natural_input).tolist()],
                                               n_results=1)
        
        print(f"Here's the query returned context {context}\n")

        return context
