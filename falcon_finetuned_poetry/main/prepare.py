import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def create_and_prepare_model(model_id):
    """
    Creating BnB configuration for model quantization
    and defining LoRA configuration for model fine-tuning
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    print("Preparing model")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    model.config.use_cache = False

    print("Preparing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Preparing config")
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
    )

    return model, tokenizer, peft_config
