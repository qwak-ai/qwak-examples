import torch


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def perform_training_cycle(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for i, data in enumerate(loader, 0):
        print("Training round", i)
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


def perform_validation_cycle(epoch, tokenizer, model, device, loader):
    """
    Function to evaluate model for predictions

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
