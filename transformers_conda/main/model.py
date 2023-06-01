import pandas as pd
import numpy as np
import qwak
import evaluate
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, ModelSchema
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

class HuggingFaceTokenizerModel(QwakModel):

    def __init__(self):
        model_id = "distilbert-base-uncased"                
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

    def build(self):
        """
        The build() method is called once during the remote build process on Qwak.
        We use it to train the model on the Yelp dataset
        """

        def tokenize(examples):
            return self.tokenizer(examples['text'],
                                  padding='max_length',
                                  truncation=True)

        dataset = load_dataset('yelp_polarity')

        print('Tokenizing dataset...')
        tokenized_dataset = dataset.map(tokenize, batched=True)

        print('Splitting data to training and evaluation sets')
        train_dataset = tokenized_dataset['train'].shuffle(seed=42).select(range(50))
        eval_dataset = tokenized_dataset['test'].shuffle(seed=42).select(range(50))

        # We don't need the tokenized dataset
        del tokenized_dataset
        del dataset

        # Defining parameters for the training process
        metric = evaluate.load('accuracy')

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

        print('Training the model...')
        trainer.train()

        # Evaluate on the validation dataset
        eval_output = trainer.evaluate()

        # Extract the validation accuracy from the evaluation metrics
        eval_acc = eval_output['eval_accuracy']

        # Log metrics into Qwak
        qwak.log_metric({"val_accuracy" : eval_acc})


    def schema(self):
        """
        schema() define the model input structure.
        Use it to enforce the structure of incoming requests.
        """
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="text", type=str),
            ])
        return model_schema
    
    @qwak.api()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The predict() method takes a pandas DataFrame (df) as input
        and returns a pandas DataFrame with the prediction output.
        """
        input_data = list(df['text'].values)
        
        # Tokenize the input data using a pre-trained tokenizer
        tokenized = self.tokenizer(input_data,
                                   padding='max_length',
                                   truncation=True,
                                   return_tensors='pt')
        
        response = self.model(**tokenized)

        return pd.DataFrame(
            response.logits.softmax(dim=1).tolist()
        )
