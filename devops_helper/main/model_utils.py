import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
from qwak.clients.secret_service import SecretServiceClient
import main.config as config


def get_hardware_config():
    """
    Determines the appropriate model configuration based on available hardware.
    This centralized function prevents code duplication.

    Returns:
        A dictionary containing quantization_config, torch_dtype, and device_map_arg.
    """
    quantization_config = None
    torch_dtype = torch.float32
    device_map_arg = None

    if torch.cuda.is_available():
        print("✅ CUDA is available. Configuring for 4-bit quantization.")
        quantization_config = config.BNB_CONFIG
        torch_dtype = config.BNB_CONFIG.bnb_4bit_compute_dtype
        device_map_arg = {"": 0}
    elif torch.backends.mps.is_available():
        print("✅ MPS is available. Loading model in bfloat16 for stability.")
        torch_dtype = torch.bfloat16
        device_map_arg = None # Let the Trainer handle device placement
    else:
        print("⚠️ No GPU detected. Loading model in float16 for CPU.")
        torch_dtype = torch.float16
        device_map_arg = None
    
    return {
        "quantization_config": quantization_config,
        "torch_dtype": torch_dtype,
        "device_map_arg": device_map_arg
    }


def login_to_hf():
    """
    Logs into Hugging Face using a token from Qwak's Secret Service.
    """
    try:
        secret_service = SecretServiceClient()
        hf_token = secret_service.get_secret("hugging-face")
        login(token=hf_token)
        print("Successfully logged into Hugging Face Hub.")
    except Exception as e:
        print(f"Error logging into Hugging Face: {e}")
        raise


def get_tokenizer(model_id: str):
    """
    Loads and configures the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model(model_id: str):
    """
    Loads the model using the centralized hardware configuration.
    """
    # Get hardware-specific settings from the new utility function
    hw_config = get_hardware_config()
    
    # Load the model's configuration first to modify it.
    model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # This is necessary for gradient checkpointing to work correctly with PEFT.
    model_config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=model_config,
        quantization_config=hw_config["quantization_config"],
        torch_dtype=hw_config["torch_dtype"],
        trust_remote_code=True,
        device_map=hw_config["device_map_arg"],
    )
    
    if hw_config["quantization_config"] is not None:
        model = prepare_model_for_kbit_training(model)
    
    peft_model = get_peft_model(model, config.LORA_CONFIG)
    peft_model.print_trainable_parameters()
    
    return peft_model
