import torch
import os
from transformers import BitsAndBytesConfig
from peft import LoraConfig

# --- Model Configuration ---
# Updated to use the Llama 2 7B Chat model
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
#MODEL_ID = "google/gemma-2b-it"

# --- Training Hyperparameters ---
TRAIN_BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE', 2))
TRAINING_EPOCHS = int(os.environ.get('TRAINING_EPOCHS', 1))
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
MAX_SEQ_LENGTH = 256

# --- Dataset Configuration ---
DATASET_ID = "stanfordnlp/imdb"
DATASET_SAMPLE_PERCENTAGE = 1

# --- LoRA (PEFT) Configuration ---
# These target modules are also correct for Llama 2 models.
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- Quantization Configuration ---
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# LLama2 (on GPU)
# --- Prompt Engineering ---
# --- FIX: Updated prompt format for Llama 2 Chat models ---
def get_prompt(text: str) -> str:
    
    # Creates a formatted prompt for the Llama 2 Chat model.
    
    system_prompt = "You are a helpful assistant that analyzes the sentiment of movie reviews."
    instruction = f"Analyze the sentiment of this movie review and classify it as 'positive' or 'negative'.\n\nReview: \"{text}\""
    
    # Llama 2 uses a specific instruction format with [INST] and <<SYS>> tags.
    return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST]"

"""
# Gemma (locally)
def get_prompt(text: str) -> str:
    
    #Creates a formatted prompt for the Gemma Instruct model.
    
    instruction = f"Analyze the sentiment of this movie review and classify it as 'positive' or 'negative'.\n\nReview: \"{text}\""
    
    # Gemma uses a specific turn-based format.
    return f"<bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"

"""