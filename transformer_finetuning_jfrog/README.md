# Movie Review Generation with Fine-tuned GPT-2 and JFrog ML

## Overview

This project demonstrates advanced text generation using a fine-tuned GPT-2 model with [JFrog ML's Machine Learning Platform](https://jfrog.com/start-free/). The model is fine-tuned on the IMDb movie reviews dataset to generate movie review-style text, showcasing sophisticated NLP techniques including model fine-tuning, version management, and deployment strategies.

### Features

- **GPT-2 Fine-tuning**: Fine-tunes GPT-2 on IMDb movie reviews for domain-specific text generation
- **Large Dataset Training**: Trains on 25,000 movie reviews with configurable data percentage
- **Model Versioning**: Advanced version management with training and deployment version separation
- **JFrog ML Integration**: Full integration with JFrog ML platform for model management and deployment
- **Hardware Optimization**: Automatic detection and utilization of GPU/MPS/CPU
- **Flexible Configuration**: Configurable training parameters via environment variables

### Key Capabilities

- Movie review-style text generation
- Configurable text length and generation parameters
- Efficient tokenization and data processing
- Cross-platform training (macOS/Linux)
- Integration with JFrog ML's model registry and versioning

---

## Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/jfrog/jfrog-ml-examples.git
   cd jfrog-ml-examples/transformer_finetuning_jfrog
   ```

2. **Install Dependencies**
   Make sure you have the required dependencies installed:
   ```bash
   pip install torch transformers datasets pandas numpy frogml
   ```

3. **Install and Configure the JFrog ML SDK**
   Use your account [JFrog ML API Key](https://docs.jfrog.com/jfrog-ml/getting-started) to set up your SDK locally:
   ```bash
   pip install qwak-sdk
   qwak configure
   ```

4. **Run the Model Locally** (Pre-trained GPT-2)
   Execute the following command to test the model locally:
   ```bash
   python test_model_locally.py
   ```

---

## Training the Model

To fine-tune the model on IMDb movie reviews:

1. **Set Training Environment Variables**
   ```bash
   export TRAIN=true
   export MODEL_VERSION=v1.0.0
   export TRAIN_BATCH_SIZE=4
   export TRAINING_EPOCHS=1
   ```

2. **Build and Train on JFrog ML Platform**
   ```bash
   qwak models create "Movie Review Generator" --project "Sample Project"
   
   qwak models build \
   --model-id <your-model-id> \
   ./transformer_finetuning_jfrog \
   --instance medium \
   -E TRAIN=true \
   -E MODEL_VERSION=v1.0.0 \
   -E TRAIN_BATCH_SIZE=4 \
   -E TRAINING_EPOCHS=1
   ```

### Training Configuration

The training process supports several configuration options:

- **Dataset Size**: Configurable percentage of IMDb dataset (default: 10%)
- **Batch Size**: Configurable training batch size (default: 4)
- **Epochs**: Number of training epochs (default: 1)
- **Learning Rate**: 5e-5 with weight decay 0.01
- **Evaluation Strategy**: Per-step evaluation every 100 steps

---

## How to Run Remotely on JFrog ML

1. **Build on the JFrog ML Platform** (without training)

   Create a new model on JFrog ML using the command:
   ```bash
   qwak models create "Movie Review Generator" --project "Sample Project"
   ```

   Build the model:
   ```bash
   qwak models build \
   --model-id <your-model-id> \
   ./transformer_finetuning_jfrog \
   --instance small
   ```

2. **Deploy the Model on the JFrog ML Platform with a Real-Time Endpoint**

   To deploy your model via the CLI, use the following command:
   ```bash
   qwak models deploy realtime \
   --model-id <your-model-id> \
   --build-id <your-build-id> \
   --instance small \
   --server-workers 1
   ```

3. **Test the Live Model with a Sample Request**

   Install the JFrog ML Inference SDK:
   ```bash
   pip install qwak-inference
   ```

   Call the Real-Time endpoint using your Model ID from the JFrog ML platform:
   ```bash
   python test_live_model.py <your-qwak-model-id>
   ```

---

## Model Architecture

The model uses **GPT-2**, a transformer-based language model designed for text generation:

- **Base Model**: `gpt2` (124M parameters)
- **Fine-tuning Dataset**: IMDb movie reviews (25,000 reviews)
- **Training Configuration**: Configurable epochs, batch size, and learning rate
- **Text Generation**: Causal language modeling with configurable output length
- **Tokenization**: GPT-2 tokenizer with EOS token as padding token

### Training Details

The fine-tuning process includes:
- **Data Preprocessing**: Tokenization with truncation and padding to 128 tokens
- **Data Collator**: Language modeling data collator for causal LM
- **Training Arguments**: Optimized for efficiency with mixed precision support
- **Model Saving**: Automatic model and tokenizer saving to JFrog ML with versioning

---

## Version Management

The project supports sophisticated version management:

### Training Version vs. Deployment Version

- **Training Version**: Set via `MODEL_VERSION` environment variable during training
- **Deployment Version**: Can be different from training version, set via `MODEL_VERSION` at runtime
- **Version Loading**: Automatically loads the specified version from JFrog ML registry

### Example Usage

```bash
# Train with version v1.0.0
export TRAIN=true
export MODEL_VERSION=v1.0.0

# Deploy with a different version (e.g., v0.9.0)
export MODEL_VERSION=v0.9.0
```

---

## Input/Output Format

### Input Schema
- **Field**: `prompt` (string)
- **Description**: Text prompt to continue/generate from
- **Example**: "This movie was absolutely"

### Output Format
- **Type**: Pandas DataFrame
- **Columns**:
  - `generated_text`: Complete generated text including the original prompt

### Example Usage

```python
import pandas as pd

# Input data
input_df = pd.DataFrame({
    'prompt': ['This movie was absolutely']
})

# Output will be:
# generated_text: 'This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout...'
```

---

## Hardware Optimization

The model automatically detects and utilizes available hardware:

- **CUDA Support**: Automatic GPU detection and utilization
- **MPS Support**: Apple Silicon GPU acceleration (Metal Performance Shaders)
- **CPU Fallback**: Graceful fallback to CPU when GPU unavailable
- **Memory Monitoring**: CUDA memory usage tracking and reporting

### Device Selection Priority
1. **MPS** (Apple Silicon GPU)
2. **CUDA** (NVIDIA GPU)  
3. **CPU** (Fallback)

---

## Project Structure

```bash
.
├── main/                           # Main directory containing core code
│   ├── __init__.py                # Package initialization
│   └── model.py                   # Movie review generation model implementation
├── jfrog_ml_demo.ipynb            # Jupyter notebook for experimentation
├── test_model_locally.py          # Script to test the model locally
├── test_live_model.py             # Script to test the live model
└── README.md                      # Documentation
```

---

## Environment Variables

The model supports several environment variables for configuration:

### Training Configuration
- **`TRAIN`**: Set to `true` to enable training mode
- **`MODEL_VERSION`**: Version identifier for the trained model (required for training)
- **`TRAIN_BATCH_SIZE`**: Training batch size (default: 4)
- **`TRAINING_EPOCHS`**: Number of training epochs (default: 1)

### Runtime Configuration
- **`MODEL_VERSION`**: Version to load for inference (can differ from training version)
- **`QWAK_MODEL_ID`**: JFrog ML model identifier
- **`QWAK_BUILD_ID`**: JFrog ML build identifier

---

## Advanced Features

### Cross-Platform Training

The model automatically adapts to different platforms:
- **macOS**: Uses current directory for output
- **Linux/Cloud**: Uses `/qwak/model_dir/` for training outputs

### Error Handling

Comprehensive error handling includes:
- **Version Validation**: Ensures MODEL_VERSION is provided when training
- **Model Loading**: Robust loading from JFrog ML with fallback to pre-trained
- **Hardware Detection**: Graceful fallback between GPU/CPU
- **Generation Safety**: Safe text generation with error handling

### Performance Optimization

- **Mixed Precision**: FP16 training support for faster training
- **Gradient Accumulation**: Efficient memory usage during training
- **Tokenization Efficiency**: Optimized tokenization with proper padding
- **Pipeline Integration**: Hugging Face pipeline for efficient inference

---

## Dataset Information

### IMDb Movie Reviews Dataset

- **Source**: Stanford Large Movie Review Dataset
- **Size**: 50,000 reviews (25,000 training, 25,000 test)
- **Configurable Subset**: Default uses 10% of dataset for efficiency
- **Preprocessing**: Automatic tokenization and truncation to 128 tokens
- **Labels**: Binary sentiment labels (removed during training for language modeling)

---

## Try JFrog ML's MLOps Platform for Free

Are you looking to deploy your machine learning models in a production-ready environment within minutes? [JFrog ML](https://jfrog.com/start-free/) offers a seamless platform to build, train, and deploy your models with ease.

Whether you're a data scientist, ML engineer, or developer, JFrog ML provides the tools and support to take your models from development to deployment effortlessly. Explore the platform and start deploying your models today. [Try JFrog ML for free!](https://jfrog.com/start-free/)