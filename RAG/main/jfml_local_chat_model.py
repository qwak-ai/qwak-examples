import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Optional, Any, Mapping, Dict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration, Generation

class JFMLLocalChatModel(BaseChatModel):
    """
    A custom LangChain BaseChatModel implementation for running Hugging Face LLMs
    locally without needing a separate web server.
    It loads the model and tokenizer directly using the transformers library.
    """
    # --- Pydantic Fields (Class Attributes) ---
    # These define the configuration and state of your model.
    # Pydantic will populate these based on kwargs or their default values.

    # model and tokenizer instances are loaded *after* Pydantic's init,
    # so they must be Optional and default to None initially.
    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None

    # Configuration parameters for the LLM.
    # Provide default values here for parameters you don't always want to require.
    # 'model_name' is typically mandatory, so it has no default here.
    model_name: str 
    max_new_tokens: int = 512 # Default value
    temperature: float = 0.7 # Default value
    top_p: float = 0.95 # Default value
    repetition_penalty: float = 1.1 # Default value
    use_chat_template: bool = True # Default value
    quantization_config: Optional[BitsAndBytesConfig] = None # Default value (no quantization)

    device: str = "auto" # Default to "auto" for dynamic detection

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # 3. Handle dynamic device determination (if 'device' was set to "auto" or similar)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Ensure device is a valid torch device string
        elif self.device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device specified: {self.device}. Must be 'cpu', 'cuda', or 'auto'.")


        print(f"Loading model '{self.model_name}' on device: {self.device}")
        print(f"Generation parameters: max_new_tokens={self.max_new_tokens}, temperature={self.temperature}, top_p={self.top_p}, repetition_penalty={self.repetition_penalty}")

        try:
            # 4. Load the tokenizer using the populated self.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # 5. Prepare loading arguments for the model
            load_kwargs = {}
            if self.quantization_config:
                load_kwargs['quantization_config'] = self.quantization_config
            elif self.device == "cuda":
                # Use bfloat16 if supported, otherwise float16 for CUDA
                load_kwargs['torch_dtype'] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
            # Set device_map based on device and quantization status
            if self.device == "cuda" and not self.quantization_config:
                load_kwargs['device_map'] = "auto"
            elif self.device == "cpu":
                load_kwargs['device_map'] = "cpu"
            # If quantization_config is present, device_map might be handled internally by BitsAndBytes,
            # or you might explicitly add 'device_map="auto"' if needed for specific setups.

            # 6. Load the model using the populated self.model_name and other attributes
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs,
                trust_remote_code=True,
            )
            self.model.eval() # Set model to evaluation mode
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            raise # Re-raise the exception after printing

    @property
    def _llm_type(self) -> str:
        return "huggingface_local"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "use_chat_template": self.use_chat_template,
        }

    def _format_messages_for_model(self, messages: List[BaseMessage]) -> str:
        if self.use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                hf_messages = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        hf_messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        hf_messages.append({"role": "assistant", "content": msg.content})
                    elif isinstance(msg, SystemMessage):
                        hf_messages.append({"role": "system", "content": msg.content})
                    else:
                        hf_messages.append({"role": "user", "content": msg.content})

                return self.tokenizer.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                print(f"Warning: Failed to apply chat template. Falling back to simple formatting. Error: {e}")
                pass

        prompt_parts = []
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                prompt_parts.append(f"AI: {message.content}")
            else:
                prompt_parts.append(f"{type(message).__name__}: {message.content}")
        
        final_prompt = "\n".join(prompt_parts) + "\nAI:"
        return final_prompt

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._format_messages_for_model(messages)


        # --- START DEBUGGING PRINTS ---
        print("\n--- Formatted Prompt String (Input to Hugging Face Tokenizer) ---")
        print(prompt)
        print("------------------------------------------------------------------")
        # --- END DEBUGGING PRINTS ---


        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        generation_config.update(kwargs)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_config)

        generated_text_ids = output_ids[0][inputs.input_ids.shape[-1]:]
        generated_text = self.tokenizer.decode(generated_text_ids, skip_special_tokens=True)

        if stop:
            for s in stop:
                if s in generated_text:
                    generated_text = generated_text.split(s)[0]
                    break
        
        if generated_text.strip().startswith("AI:"):
            generated_text = generated_text.strip()[3:].strip()
        
        message = AIMessage(content=generated_text)
        generation = ChatGeneration(message=message, text=generated_text)
        
        return ChatResult(generations=[generation])
