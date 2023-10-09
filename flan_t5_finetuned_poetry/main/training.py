import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from main.helpers import get_device
from main.dataset_loader import TextDataset


def perform_training_cycle(epoch: int,
                           tokenizer: T5Tokenizer,
                           model: T5ForConditionalGeneration,
                           device: torch.device,
                           loader: DataLoader,
                           optimizer: AdamW
                           ):
    """
    Run a single training epoch
    """
    model.train()
    for i, data in enumerate(loader, 0):
        print("Training epoch", epoch, "batch", i)
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def perform_validation_cycle(epoch: int,
                             tokenizer: T5Tokenizer,
                             model: T5ForConditionalGeneration,
                             device: torch.device,
                             loader: DataLoader
                             ):
    """
    Evaluate model for predictions
    """
    model.eval()
    predictions = []
    actual_results = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            model_predictions = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in generated_ids
            ]
            target = [
                tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for t in y
            ]

            predictions.extend(model_predictions)
            actual_results.extend(target)
    return predictions, actual_results


def train_model(dataframe: DataFrame,
                source_text: str,
                target_text: str,
                model_params: Dict,
                output_dir: str = "./outputs/"
                ):
    """
    Perform full model training cycle
    """

    # Set random seeds for numpy and pytorch for reproducibility
    torch.manual_seed(model_params["seed"])
    np.random.seed(model_params["seed"])
    torch.backends.cudnn.deterministic = True

    # Tokenizer for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(
        model_params["model_id"],
        model_max_length=model_params["max_source_text_length"]
    )

    device = get_device()
    print(f"Using device: {device}")
    model = T5ForConditionalGeneration.from_pretrained(model_params["model_id"])
    model = model.to(device)

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]

    # Defining the train size, 80% of the data will be used for training and the rest for validation.
    train_size = 0.8
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["seed"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = TextDataset(
        train_dataset,
        tokenizer,
        model_params["max_source_text_length"],
        model_params["max_target_text_length"],
        source_text,
        target_text,
    )

    val_set = TextDataset(
        val_dataset,
        tokenizer,
        model_params["max_source_text_length"],
        model_params["max_target_text_length"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["train_batch_size"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["valid_batch_size"],
        "shuffle": False,
        "num_workers": 0,
    }

    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=model_params["learning_rate"]
    )

    for epoch in range(model_params["train_epochs"]):
        print("Training cycle started")
        perform_training_cycle(epoch, tokenizer, model, device, training_loader, optimizer)
        print("Training cycle ended")

    # Saving the model after training
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    # evaluating test dataset
    for epoch in range(model_params["val_epochs"]):
        predictions, actual_results = perform_validation_cycle(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({
            "Generated Text": predictions,
            "Actual Text": actual_results
        })
        final_df.to_csv(os.path.join(output_dir, "predictions.csv"))

    return model