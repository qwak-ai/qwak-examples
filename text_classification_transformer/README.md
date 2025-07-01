# Text Classification with DistilBERT and JFrog ML

## Overview

This project demonstrates text classification using a fine-tuned DistilBERT model with [JFrog ML's Machine Learning Platform](https://jfrog.com/start-free/). The model is trained on the AG News dataset for news article classification and showcases advanced NLP techniques including fine-tuning, analytics logging, and model versioning.

### Features

- **DistilBERT Fine-tuning**: Fine-tunes DistilBERT on AG News dataset for news classification
- **4-Class Classification**: Classifies news articles into World, Sports, Business, and Technology categories
- **Advanced Analytics**: Logs sentence metrics including length, word count, and spaces
- **JFrog ML Integration**: Full integration with JFrog ML platform for model management and deployment
- **GPU/MPS Support**: Automatic detection and utilization of available hardware acceleration
- **Model Versioning**: Supports training and deployment with different model versions

### Key Capabilities

- News article classification with confidence scores
- Real-time analytics logging during inference
- Efficient tokenization and batch processing
- Hardware-optimized training and inference
- Integration with JFrog ML's model registry

---

## Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/jfrog/jfrog-ml-examples.git
   cd jfrog-ml-examples/text_classification_transformer
   ```

2. **Install Dependencies**
   Make sure you have the required dependencies installed:
   ```bash
   pip install torch transformers datasets evaluate pandas numpy scikit-learn frogml
   ```

3. **Install and Configure the JFrog ML SDK**
   Use your account [JFrog ML API Key](https://docs.jfrog.com/jfrog-ml/getting-started) to set up your SDK locally:
   ```bash
   pip install qwak-sdk
   qwak configure
   ```

4. **Run the Model Locally**
   Execute the following command to test the model locally:
   ```bash
   python test_model_locally.py
   ```

---

## Training the Model

To train the model with fine-tuning:

1. **Set Training Environment Variables**
   ```bash
   export TRAIN=true
   export MODEL_VERSION=v1.0.0
   ```

2. **Build and Train on JFrog ML Platform**
   ```bash
   qwak models create "Text Classification" --project "Sample Project"
   
   qwak models build \
   --model-id <your-model-id> \
   ./text_classification_transformer \
   --instance medium \
   -E TRAIN=true \
   -E MODEL_VERSION=v1.0.0
   ```

---

## How to Run Remotely on JFrog ML

1. **Build on the JFrog ML Platform** (without training)

   Create a new model on JFrog ML using the command:
   ```bash
   qwak models create "Text Classification" --project "Sample Project"
   ```

   Build the model:
   ```bash
   qwak models build \
   --model-id <your-model-id> \
   ./text_classification_transformer \
   --instance small \
   -E MODEL_VERSION=v1.0.0
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

The model uses **DistilBERT**, a lightweight version of BERT designed for efficient text classification:

- **Base Model**: `distilbert-base-uncased`
- **Fine-tuning Dataset**: AG News (1000 training samples, 500 evaluation samples)
- **Classes**: 4 categories (World, Sports, Business, Technology)
- **Training Configuration**: 3 epochs, learning rate 2e-5, batch size 4
- **Evaluation Metric**: Accuracy

### Training Details

The fine-tuning process includes:
- **Data Preprocessing**: Tokenization with truncation and padding
- **Training Arguments**: Optimized for efficiency and performance
- **Evaluation Strategy**: Per-epoch evaluation with accuracy metrics
- **Model Saving**: Automatic model and tokenizer saving to JFrog ML

---

## Input/Output Format

### Input Schema
- **Field**: `text` (string)
- **Description**: Raw text to be classified
- **Example**: "Apple Inc. reported strong quarterly earnings..."

### Output Format
- **Type**: Pandas DataFrame
- **Columns**:
  - `label`: Predicted class name
  - `score`: Confidence score for the prediction

### Example Usage

```python
import pandas as pd

# Input data
input_df = pd.DataFrame({
    'text': ['Apple Inc. reported strong quarterly earnings this quarter.']
})

# Output will be:
# label: 'Business'
# score: 0.95
```

---

## Analytics and Monitoring

The model includes comprehensive analytics logging:

- **Sentence Length**: Character count of input text
- **Word Count**: Number of words in the input
- **Space Count**: Number of spaces in the input
- **Real-time Logging**: All metrics logged during inference

### Analytics Data

During inference, the following metrics are automatically logged:
```python
{
    "sentence_length": "123",
    "num_spaces": "15", 
    "num_words": "16"
}
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
│   └── model.py                   # Text classification model implementation
├── experimentation.ipynb          # Jupyter notebook for experimentation
├── test_model_locally.py          # Script to test the model locally
├── test_live_model.py             # Script to test the live model
└── README.md                      # Documentation
```

---

## Environment Variables

The model supports several environment variables for configuration:

- **`TRAIN`**: Set to `true` to enable training mode
- **`MODEL_VERSION`**: Version identifier for the trained model
- **`QWAK_MODEL_ID`**: JFrog ML model identifier
- **`QWAK_BUILD_ID`**: JFrog ML build identifier

---

## Performance Considerations

- **Model Size**: DistilBERT is 40% smaller than BERT-base
- **Inference Speed**: Optimized for real-time classification
- **Memory Usage**: Efficient memory management with automatic cleanup
- **Batch Processing**: Supports efficient batch tokenization
- **Hardware Acceleration**: Automatic GPU utilization when available

---

## Error Handling

The model includes comprehensive error handling:
- **Hardware Detection**: Graceful fallback between GPU/CPU
- **Model Loading**: Robust model and tokenizer loading from JFrog ML
- **Input Validation**: Validates input format and schema
- **Training Monitoring**: Comprehensive logging during training

---

## Try JFrog ML's MLOps Platform for Free

Are you looking to deploy your machine learning models in a production-ready environment within minutes? [JFrog ML](https://jfrog.com/start-free/) offers a seamless platform to build, train, and deploy your models with ease.

Whether you're a data scientist, ML engineer, or developer, JFrog ML provides the tools and support to take your models from development to deployment effortlessly. Explore the platform and start deploying your models today. [Try JFrog ML for free!](https://jfrog.com/start-free/)