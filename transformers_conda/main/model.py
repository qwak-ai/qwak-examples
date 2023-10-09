# Import required libraries
import evaluate
import numpy as np
import pandas as pd
import qwak
from datasets import load_dataset
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, ModelSchema
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch

# Define the model class
class HuggingFaceTokenizerModel(QwakModel):

    def __init__(self):
        # Initialize tokenizer and model
        model_id = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
        self.device = torch.device("cpu")

        # Device configuration (MPS for MacOS, otherwise CUDA or CPU)
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ or no MPS-enabled device.")
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("mps")


    def build(self):
        """
        Train the model on the Yelp dataset. Called once during the remote build process on Qwak.
        """
        # Tokenization function
        def tokenize(examples):
            return self.tokenizer(examples['text'], padding='max_length', truncation=True)

        # Load and tokenize dataset
        dataset = load_dataset('yelp_polarity')
        tokenized_dataset = dataset.map(tokenize, batched=True)

        # Split dataset into training and evaluation sets
        train_dataset = tokenized_dataset['train'].shuffle(seed=42).select(range(50))
        eval_dataset = tokenized_dataset['test'].shuffle(seed=42).select(range(50))

        # Cleanup
        del tokenized_dataset
        del dataset

        # Define training arguments and metrics
        metric = evaluate.load('accuracy')

        # A helper method to evaluate the model during training
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=1)
            return metric.compute(predictions=predictions, references=labels)
        
        training_args = TrainingArguments(output_dir='training_output', evaluation_strategy='epoch', num_train_epochs=1)

        # Initialize Trainer
        trainer = Trainer(model=self.model, 
                          args=training_args, 
                          train_dataset=train_dataset, 
                          eval_dataset=eval_dataset,
                          compute_metrics=compute_metrics)

        # Train the model
        trainer.train()

        # Evaluate and log metrics
        eval_output = trainer.evaluate()
        qwak.log_metric({"eval_accuracy": eval_output['eval_accuracy']})


    def schema(self):
        """
        Define the model input schema.
        """
        return ModelSchema(inputs=[ExplicitFeature(name="text", type=str)])


    @qwak.api()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on input DataFrame and return output as DataFrame.
        """
        # Extract text data and tokenize
        input_data = list(df['text'].values)
        tokenized = self.tokenizer(input_data, padding='max_length', truncation=True, return_tensors='pt')

        # Move tensors to the configured device
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        # Make predictions
        response = self.model(**tokenized)

        # Return softmax probabilities as DataFrame
        return pd.DataFrame(response.logits.softmax(dim=1).tolist())
