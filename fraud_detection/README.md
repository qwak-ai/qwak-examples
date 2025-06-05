# XGBoost Churn Prediction Model with Qwak

## Overview

This project employs the XGBoost algorithm for customer churn prediction. It's implemented using the [Qwak's Machine Learning Platform](https://www.qwak.com/) and the XGBoost library.

### Features

- **Custom XGBoostChurnPredictionModel Class Definition**: Customizes the base QwakModel to work with the XGBoost algorithm for churn prediction.

- **Model Initialization**: Initializes the XGBoost model with user-defined or default hyperparameters. The model is trained on a customer churn dataset and fine-tuned for optimal performance.

- **Churn Prediction via Qwak's Predict API**: Utilizes Qwak's Predict API for assessing the probability of customer churn based on various features like account length, area code, international plan, etc.

### Functionality

The primary functionality is to predict the probability of customer churn. The code is designed for seamless integration with Qwak's platform and serves as a practical example for churn prediction tasks.



<br>

## How to Test Locally


1. **Clone the Repository**: Clone this GitHub repository to your local machine.

2. **Install Dependencies**: Make sure you have the required dependencies installed, as specified in the `conda.yml` file.

    ```bash
    conda env create -f main/conda.yaml
    conda activate churn_model
    ```

3. **Install and Configure the Qwak SDK**: Use your account [Qwak API Key](https://docs.qwak.com/docs/getting-started#configuring-qwak-sdk) to set up your SDK locally.

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
    qwak models create "Churn Prediction Model" --project "Sample Project"
    ```


    Initiate a model build with:

    ```bash
    qwak models build --model-id <your-model-id> ./churn_model_new
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
│   ├── model.py           # Defines the Churn Model
│   ├── data.csv           # Defines the data to train the Model
│   └── conda.yaml         # Conda environment configurationdata
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
