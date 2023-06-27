import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from prepare import create_and_prepare_model


def train_model(model_id):
    """
    The entire training flow with PEFT using LoRA and BnB 4 bit quantization
    """

    training_arguments = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=10,
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=500,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
    )

    print("Creating and preparing model")
    model, tokenizer, peft_config = create_and_prepare_model(model_id)

    print("Loading dataset")
    dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

    print("Preparing SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments
    )

    # Pre-process the model by upcasting the layer norms in float 32 for more stable training
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    print("Starting model trainer")
    trainer.train()

    return model, tokenizer
