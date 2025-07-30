import platform
import torch
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, ModelSchema
from qwak.model.adapters import JsonInputAdapter, JsonOutputAdapter
import qwak
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import PeftModel, PeftConfig

# Import our refactored utilities and configuration
import main.config as config
import main.data_utils as data_utils
import main.model_utils as model_utils

class LLMFineTuner(QwakModel):
    """
    A QwakModel class to fine-tune and serve a Llama2 8B model.

    This class is a high-level orchestrator, delegating tasks
    to helper modules. It defines the model's lifecycle for the Qwak platform.
    """

    def __init__(self):
        """
        Initializes paths and placeholders. No heavy objects are stored here
        to keep the initial object light before pickling.
        """
        self.generator = None
        self.model = None
        self.tokenizer = None
        
        # Define the directory where the fine-tuned adapter will be saved.
        if platform.system() == 'Darwin': # For local testing
            self.adapter_output_dir = './llama-imdb-finetuned'
        else: # For the Qwak environment
            self.adapter_output_dir = "/qwak/model_dir/llama-imdb-finetuned"


    def build(self):
        """
        The main training logic. This method is called by Qwak to build the model.
        It trains the model and saves the resulting LoRA adapter and tokenizer to disk.
        """
        print(f"Starting build process for model: {config.MODEL_ID}")
        model_utils.login_to_hf()

        # 1. Load Tokenizer and Model using our updated utility functions
        tokenizer = model_utils.get_tokenizer(config.MODEL_ID)

        # Using left-padding with the beginning-of-sentence token is more robust.
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.bos_token


        model = model_utils.get_model(config.MODEL_ID)

        # 2. Load and Tokenize Dataset
        tokenized_dataset = data_utils.load_and_tokenize_dataset(
            tokenizer,
            percentage=config.DATASET_SAMPLE_PERCENTAGE,
            max_length=config.MAX_SEQ_LENGTH
        )

        # 3. Configure and run the Hugging Face Trainer
        use_fp16 = False
        use_bf16 = False
        if torch.cuda.is_available():
            print("✅ CUDA detected. Configuring mixed precision.")
            if torch.cuda.is_bf16_supported():
                use_bf16 = True
            else:
                use_fp16 = True
        else:
            print("⚠️ No CUDA detected. Mixed precision flags (fp16/bf16) will be disabled.")


        training_args = TrainingArguments(
            output_dir=self.adapter_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=config.TRAINING_EPOCHS,
            per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=config.TRAIN_BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            bf16=use_bf16,
            fp16=use_fp16,
            logging_steps=10,
            save_total_limit=1,
            eval_strategy="steps",
            eval_steps=100,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
        )

        # 4. Start Training and Log Artifacts
        print("Starting model training...")
        train_output = trainer.train()
        print("Training complete.")
        
        # --- ADDITION: Log parameters and metrics ---
        print("Logging parameters and metrics...")

        # Log all hyperparameters used for the training run
        params_to_log = {
            "model_id": config.MODEL_ID,
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.TRAINING_EPOCHS,
            "batch_size": config.TRAIN_BATCH_SIZE,
            "max_seq_length": config.MAX_SEQ_LENGTH,
            "lora_r": config.LORA_CONFIG.r,
            "lora_alpha": config.LORA_CONFIG.lora_alpha,
            "lora_target_modules": str(config.LORA_CONFIG.target_modules),
        }
        qwak.log_param(params_to_log)

        # Log the final metrics from the training process
        final_metrics = train_output.metrics
        
        # Find the last evaluation loss from the history
        eval_logs = [log for log in trainer.state.log_history if 'eval_loss' in log]
        if eval_logs:
            final_metrics['final_eval_loss'] = eval_logs[-1]['eval_loss']

        qwak.log_metric(final_metrics)
        print(f"Logged Parameters: {params_to_log}")
        print(f"Logged Metrics: {final_metrics}")
        # --- END ADDITION ---

        trainer.save_model(self.adapter_output_dir)
        tokenizer.save_pretrained(self.adapter_output_dir)
        print(f"Adapter and tokenizer saved to {self.adapter_output_dir}")


    def initialize_model(self):
        """
        Initializes the model for inference. It loads the base model and
        attaches the fine-tuned LoRA adapter, conditionally using quantization.
        """
        print("Initializing model for inference...")
        
        adapter_config = PeftConfig.from_pretrained(self.adapter_output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_output_dir)

        # Using left-padding with the beginning-of-sentence token is more robust for generation.
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.bos_token

        # Get hardware-specific settings from the centralized utility function
        hw_config = model_utils.get_hardware_config()
        
        # Load the base model with the appropriate config
        base_model = AutoModelForCausalLM.from_pretrained(
            adapter_config.base_model_name_or_path,
            quantization_config=hw_config["quantization_config"],
            device_map=hw_config["device_map_arg"],
            torch_dtype=hw_config["torch_dtype"],
            trust_remote_code=True,
        )
        
        # Create a PeftModel by attaching the loaded adapter to the base model.
        self.model = PeftModel.from_pretrained(base_model, self.adapter_output_dir)
        print("Successfully loaded base model and attached LoRA adapter.")

        # Explicitly move the model to MPS if not using device_map
        if not hw_config["device_map_arg"] and torch.backends.mps.is_available():
            self.model.to("mps")
            print("Model explicitly moved to MPS device.")

        # Let the pipeline figure out the device from the loaded model
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )


    def schema(self):
        """
        Defines the input schema for the model's prediction API.
        """
        return ModelSchema(inputs=[ExplicitFeature(name="prompt", type=str)])

    @qwak.api(input_adapter=JsonInputAdapter(), output_adapter=JsonOutputAdapter())
    def predict(self, json_objects):
        """
        The main inference method. It takes a list of JSON objects and returns predictions.
        """
        prompts = [data['prompt'] for data in json_objects]
        formatted_prompts = [config.get_prompt(p) for p in prompts]

        outputs = self.generator(
            formatted_prompts,
            max_new_tokens=50,
            do_sample=True,
            temperature = 0.7,
            top_k=50,
            top_p=0.95,
        )
        
        return [output[0]['generated_text'] for output in outputs]
