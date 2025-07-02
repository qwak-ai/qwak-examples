from datasets import load_dataset
from functools import partial
import main.config as config

def load_and_tokenize_dataset(tokenizer, percentage: int, max_length: int):
    """
    Loads the IMDb dataset, samples it, and tokenizes it for training.

    Args:
        tokenizer: The tokenizer instance to use.
        percentage (int): The percentage of the dataset to use.
        max_length (int): The maximum sequence length for tokenization.

    Returns:
        A tokenized dataset object.
    """
    dataset = load_dataset(config.DATASET_ID)

    # Shuffle and select a percentage of the data
    train_size = int(len(dataset['train']) * (percentage / 100))
    test_size = int(len(dataset['test']) * (percentage / 100))
    
    dataset['train'] = dataset['train'].shuffle(seed=42).select(range(train_size))
    dataset['test'] = dataset['test'].shuffle(seed=42).select(range(test_size))

    # Create a partial function for clean mapping
    tokenize_fn = partial(
        _tokenize_batch,
        tokenizer=tokenizer,
        max_length=max_length
    )

    # Apply tokenization to the dataset
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text", "label"],
        num_proc=4,
    )

    return tokenized_dataset

def _tokenize_batch(examples, tokenizer, max_length: int):
    """
    Helper function to tokenize a batch of examples using the prompt format.
    """
    # Apply the prompt formatting to each text example
    formatted_prompts = [config.get_prompt(text) for text in examples['text']]
    
    return tokenizer(
        formatted_prompts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
