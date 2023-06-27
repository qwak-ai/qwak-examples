from peft import PeftModel
from prepare import create_and_prepare_model


def init_pretrained_model(model_id):
    """
    Initialized the pretrained model
    """
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_use_double_quant=False,
    # )
    #
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_id,
    #     trust_remote_code=True
    # )
    # tokenizer.pad_token = tokenizer.eos_token
    #
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     quantization_config=bnb_config,
    #     device_map="auto",
    #     trust_remote_code=True
    # )

    model, tokenizer, _ = create_and_prepare_model(model_id)
    model = PeftModel.from_pretrained(model, "results/checkpoint-200/")

    return model, tokenizer
