# Image Classification Model with MobileNetV2 and JFrog ML

## Overview

This project demonstrates image classification using a quantized MobileNetV2 model with [JFrog ML's Machine Learning Platform](https://jfrog.com/start-free/). The model is optimized for embedded deployment with quantization techniques and provides ImageNet-based classification capabilities.

### Features

- **Quantized MobileNetV2 Model**: Uses a lightweight, quantized version of MobileNetV2 for efficient inference
- **ImageNet Classification**: Classifies images into 1000 ImageNet categories
- **JFrog ML Integration**: Seamlessly integrates with JFrog ML platform for model management and deployment
- **Embedded Optimization**: Optimized for deployment on resource-constrained environments
- **Preprocessing Pipeline**: Includes robust image preprocessing with normalization and resizing

### Key Capabilities

- Image classification with top-3 predictions
- Automatic model quantization for performance optimization
- Support for various image formats through numpy array input
- Integration with JFrog ML's model registry and deployment pipeline

---

## Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/jfrog/jfrog-ml-examples.git
   cd jfrog-ml-examples/image_classifier
   ```

2. **Install Dependencies**
   Make sure you have the required dependencies installed:
   ```bash
   pip install torch torchvision numpy pandas frogml
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

## How to Run Remotely on JFrog ML

1. **Build on the JFrog ML Platform**

   Create a new model on JFrog ML using the command:
   ```bash
   qwak models create "Image Classifier" --project "Sample Project"
   ```

   Build the model:
   ```bash
   qwak models build \
   --model-id <your-model-id> \
   ./image_classifier \
   --instance medium
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

The model uses **MobileNetV2**, a lightweight convolutional neural network designed for mobile and embedded applications:

- **Pre-trained Weights**: Uses ImageNet pre-trained weights
- **Quantization**: Applies post-training quantization for efficiency
- **Input Size**: 224x224 RGB images
- **Output**: Top-3 predictions with confidence scores
- **Categories**: 1000 ImageNet classes

### Quantization Details

The model applies several optimization techniques:
- **Layer Fusion**: Combines consecutive layers for better performance
- **FBGEMM Backend**: Uses Facebook's FBGEMM for x86 quantization
- **Calibration**: Performs calibration with sample data
- **INT8 Conversion**: Converts floating-point operations to INT8

---

## Input/Output Format

### Input
- **Type**: Numpy array representing an image
- **Format**: RGB image as numpy array (height, width, channels)
- **Preprocessing**: Automatic resizing to 224x224 and normalization

### Output
- **Type**: Pandas DataFrame
- **Columns**: 
  - `name`: Class name from ImageNet
  - `probability`: Confidence score (0-1)
- **Rows**: Top-3 predictions sorted by confidence

### Example Usage

```python
import numpy as np
from PIL import Image

# Load and convert image to numpy array
image = Image.open('sample_image.jpg')
img_array = np.array(image)

# The model will automatically handle preprocessing
# Output will be a DataFrame with top-3 predictions
```

---

## Project Structure

```bash
.
├── main/                           # Main directory containing core code
│   ├── __init__.py                # Package initialization
│   ├── model.py                   # Image classification model implementation
│   └── imagenet_classes.txt       # ImageNet class labels
├── embedded_code/                  # Embedded deployment code
├── tests/                          # Test files
├── test_model_locally.py          # Script to test the model locally
├── test_live_model.py             # Script to test the live model
├── model_qwak.py                  # Alternative model implementation
└── README.md                      # Documentation
```

---

## Performance Considerations

- **Quantization**: Reduces model size and improves inference speed
- **MobileNetV2**: Designed for efficiency with depthwise separable convolutions
- **Batch Processing**: Optimized for single image inference
- **Memory Usage**: Reduced memory footprint through quantization

---

## Supported Image Formats

The model supports various input formats:
- **RGB Images**: Standard 3-channel color images
- **Grayscale Conversion**: Automatically converts grayscale to RGB
- **JSON String Input**: Supports JSON-encoded image arrays
- **Numpy Arrays**: Direct numpy array input

---

## Error Handling

The model includes comprehensive error handling:
- **Quantization Fallback**: Falls back to original model if quantization fails
- **Backend Detection**: Automatically detects available quantization backends
- **Input Validation**: Validates and preprocesses various input formats
- **Logging**: Detailed logging for debugging and monitoring

---

## Try JFrog ML's MLOps Platform for Free

Are you looking to deploy your machine learning models in a production-ready environment within minutes? [JFrog ML](https://jfrog.com/start-free/) offers a seamless platform to build, train, and deploy your models with ease.

Whether you're a data scientist, ML engineer, or developer, JFrog ML provides the tools and support to take your models from development to deployment effortlessly. Explore the platform and start deploying your models today. [Try JFrog ML for free!](https://jfrog.com/start-free/)