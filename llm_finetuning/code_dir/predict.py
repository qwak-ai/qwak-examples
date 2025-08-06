from typing import Any
import torch
import pandas as pd

def get_best_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, data_input, **kwargs) -> Any:
    tokenizer = kwargs.get('tokenizer', None)


    # 1. Ensure the input DataFrame has the required column
    if 'prompt' not in data_input.columns:
        raise ValueError("Input DataFrame must contain a 'prompt' column.")
    
    prompts = data_input['prompt'].tolist()
    
    # 2. Apply the chat template for each prompt in the batch
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful DevOps assistant."},
                {"role": "user", "content": p}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        for p in prompts
    ]
    
    # 3. Tokenize the formatted prompts and move tensors to the correct device
    device = get_best_device()
    
    # 4. Tokenize the formatted prompts with correct device
    inputs = tokenizer(
        formatted_prompts, 
        return_tensors="pt", 
        padding=True
    ).to(device)
    
    # 5. Ensure the model is on the same device
    model = model.to(device)
    
    with torch.no_grad():
        # Using parameters from your snippet
        generated_ids = model.generate(
            **inputs, # Pass both input_ids and attention_mask
            max_new_tokens=256,
        )
    
    # 6. Decode the generated token IDs back to text
    decoded_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    # 7. Return the results as a new DataFrame
    return pd.DataFrame({
        'generated_text': decoded_responses
    })