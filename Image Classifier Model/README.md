# Image Classifier Model with ResNet and Qwak

## Overview

This project utilizes a specialized version of the ResNet model, known as ResNet50, for classifaction of images. It's implemented using the Qwak's Machine Learning Platform.
### Features

- **Custom ImageClassifier Class Definition**: Customizes the base QwakModel to work with the CatBoost algorithm for survival prediction.

- **Model Initialization**: Initializes the ResNet50 model with pre-trained weights.

- **Image classification via Qwak's Predict API**: Utilizes Qwak's Predict API for assessing the classification of Images.


<br>

## How to Test Locally


1. **Clone the Repository**: Clone this GitHub repository to your local machine.

2. **Install Dependencies**: Make sure you have the required dependencies installed, as specified in the `conda.yml` file.

    ```bash
    conda env create -f main/conda.yaml
    conda activate Image classifier
    ```

3. **Install and Configure the Qwak SDK**: Use your account [Qwak API Key](https://docs-saas.qwak.com/docs/getting-started#configuring-qwak-sdk) to set up your SDK locally.

    ```bash
    pip install qwak-sdk
    qwak configure
    ```

5. **Run the Model Locally**: Execute the following command to test the model locally:

   ```bash
   python test_model_locally.py
   ```

<br>

<br>

## How to Run Remotely on Qwak

1. **Build on the Qwak Platform**:

    Create a new model on Qwak using the command:

    ```bash
    qwak models create "Image Classifier Model" --project "Image Classification"
    ```


    Initiate a model build with:

    ```bash
    qwak models build --model-id <your-model-id> ./Image Classifier Model
    ```


2. **Deploy the Model on the Qwak Platform with a Real-Time Endpoint**:

    To deploy your model via the CLI, use the following command:

    ```bash
    qwak models deploy realtime --model-id <your-model-id> --build-id <your-build-id>
    ```

3. **Test the Live Model with a Sample Request**:

    Install the Qwak Inference SDK:

    ```bash
    pip install qwak-inference
    ```

    Call the Real-Time endpoint using your Model ID from the Qwak platform:

    ```bash
    python test_live_mode.py <your-qwak-model-id>
    ```

<br>


## Project Structure

```bash
.
├── main                   # Main directory containing core code
│   ├── __init__.py        # An empty file that indicates this directory is a Python package
│   ├── model.py           # Defines the Titanic Survival Model
│   └── conda.yaml         # Conda environment configuration
|   └── cat.jpeg           # Picture file to test the model
|
├── test_model_locally.py  # Script to test the model locally
├── test_live_model.py     # Script to test the live model with a sample REST request
└── README.md              # Documentation
```


<br>
<br>

## Try Qwak's MLOps Platform for Free

Are you looking to deploy your machine learning models in a production-ready environment within minutes? [Qwak](https://www.qwak.com/) offers a seamless platform to build, train, and deploy your models with ease.

Whether you're a data scientist, ML engineer, or developer, Qwak provides the tools and support to take your models from development to deployment effortlessly. Explore the platform and start deploying your models today. [Try Qwak for free!](https://www.qwak.com/)