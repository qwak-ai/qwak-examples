import pandas as pd
from frogml import FrogMlModel
from frogml_core.model.schema import ExplicitFeature, ModelSchema
from transformers import pipeline
import platform
import frogml
import os

import torch
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,)

from frogml_core.tools.logger import get_frogml_logger

logger = get_frogml_logger()

JF_REPOSITORY = 'nlp-models'
JF_MODEL = os.environ['QWAK_MODEL_ID']

class MovieReviewGenerator(FrogMlModel):

    # ----- Class Initialization -----
    def __init__(self):
        self.model_name = 'gpt2'
        self.tokenizer_name = 'gpt2'
        self.train = os.environ.get('TRAIN', None)
        self.model_version = os.environ.get('MODEL_VERSION', None)

        if self.train and not self.model_version: 
            raise Exception("If the model is set to be finetuned then a Model Version is necessary as input")

        self.batch_size = os.environ.get('TRAIN_BATCH_SIZE', 4)
        self.training_epochs = os.environ.get('TRAINING_EPOCHS', 1)
        self.tokenized_dataset = None
        self.tokenizer = None

        # Detect the OS
        if self.train:
            if platform.system() == 'Darwin':  # MacOS
                self.train_output_dir = '.'
            else:
                self.train_output_dir = "/qwak/model_dir/gpt2-imdb-finetuned"

    def _load_and_tokenize_dataset(self, percentage=1): 

        if not self.tokenized_dataset or not self.tokenizer:
            # Load IMDb dataset
            dataset = load_dataset("stanfordnlp/imdb")

            dataset['train'] = dataset['train'].shuffle(seed=42).select(range(int(25000*percentage/100)))
            dataset['test'] = dataset['test'].shuffle(seed=42).select(range(int(25000*percentage/100)))

            # Tokenize the dataset
            def tokenize_function(examples):
                return self.tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)

            self.tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"], num_proc=4)

        return self.tokenized_dataset, self.tokenizer

    
    # ----- Logs a HF model and tokenizer to the JFrog repository. -----
    def _log_model_to_jfrog(self, model, tokenizer, version ):

        try:
            frogml.huggingface.log_model(
                model = model,
                tokenizer = tokenizer,
                model_name = JF_MODEL,
                repository = JF_REPOSITORY,
                version = version
            )
            logger.info(f"Successfully logged model '{os.environ['QWAK_MODEL_ID']}' to JFrog repository '{self.jfrog_repository}'.")

        except KeyError as e:
            logger.error(f"Error: Environment variable 'QWAK_MODEL_ID' not found. Please set this variable.")

        except Exception as e:
            logger.error(f"An error occurred while logging the model to JFrog: {e}")

    
    # ----- Loads a HF model and tokenizer from a JFrog repository. -----
    def _load_model_from_jfrog(self, version):
        try:
            model, tokenizer = frogml.huggingface.load_model(
                repository=JF_REPOSITORY,
                model_name=JF_MODEL,
                version=version
            )
            return model, tokenizer
            
        except Exception as e:
            logger.exception(f"An error occurred while loading model{version} from JFrog: {e}")
            raise  # Re-raise the exception to signal a critical error
        finally:
            logger.info(f"Attempted to load FINETUNED model '{JF_MODEL}', version - {version} from JFrog repository '{JF_REPOSITORY}'.")

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

        # Load the GPT-2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.train:

            self._load_and_tokenize_dataset(percentage=10)

            # 4. Data Collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            )  # mlm=False for causal LM

            # Load a pre-trained GPT-2 model from Hugging Face
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device)

            self.model.resize_token_embeddings(len(self.tokenizer))  # Resize embeddings

            # 6. Training Arguments
            training_args = TrainingArguments(
                output_dir=self.train_output_dir,
                overwrite_output_dir=True,
                num_train_epochs=self.training_epochs,  # Reduced epochs for faster demonstration
                per_device_train_batch_size=self.batch_size,  # Adjusted batch size
                per_device_eval_batch_size=self.batch_size,
                eval_steps=100,
                save_steps=500,
                save_total_limit=1,
                evaluation_strategy="steps",
                learning_rate=5e-5,
                weight_decay=0.01,
                #fp16=True,  # Enable mixed precision for M2 (if supported)
            )

            # 7. Trainer Setup
            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=self.tokenized_dataset["train"],
                eval_dataset=self.tokenized_dataset["test"],
            )

            # Start training
            trainer.train()

            self._log_model_to_jfrog(self.model, self.tokenizer, self.model_version)


    # ----- RUNTIME INITIALIZATION LOGIC -----
    def initialize_model(self):

        runtime_version = os.getenv('MODEL_VERSION', None) ## deployment version might be different than the trainin gone, aka loading an older version of a finetuned model

        if runtime_version and runtime_version != self.model_version:
            self.model, self.tokenizer = self._load_model_from_jfrog(runtime_version) # loading the deployment version

            logger.info(f"The model is set to load a different finetuned a Model Version {runtime_version}")

        elif not self.train:
            logger.info(f'Loading from pretrained {self.model_name}, without finetuning.')
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)

        self.set_torch_device() # set the device again, just in case

        # 3. Create the Pipeline
        self.generator = pipeline(
            "text-generation",
            model = self.model,
            tokenizer = self.tokenizer,
            device = self.device,  # Specify the device for the pipeline
        )


    # ----- INPUT/OUTPUT SCHEMA ------
    def schema(self):

        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),
            ])
        return model_schema


    def _generate_text(self, prompt):

        try:
            generated_text = self.generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
            return generated_text
        except Exception as e:
            logger.error(f"Error generating text for prompt: {prompt}. Error: {e}")
            return None  # or some other placeholder


    # ----- INFERENCE LOGIC ------
    @frogml.api()
    def predict(self, df):

        df['generated_text'] = df['prompt'].apply(self._generate_text)
        df.drop('prompt', axis=1, inplace=True)

        return df
