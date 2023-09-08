# Feature Set - Quickstart Guide Project

## Overview

This project demonstrates how to extract, process, store, and consume features using [Qwak's Feature Store](https://www.qwak.com/product/feature-store). It includes examples of defining a Data Source, working with Feature Sets, and running a Credit Risk Machine Learning model. 

The code is designed to work in conjunction with the [Quickstart Guide](https://docs-saas.qwak.com/docs/getting-started-copy) provided by Qwak.


## How to Test Locally


1. **Clone the Repository**: Clone this GitHub repository to your local machine.

2. **Install Dependencies**: Make sure you have the required dependencies installed, as specified in the `conda.yml` file.

    ```bash
    conda env create -f main/conda.yaml
    conda activate feature_set_quickstart_guide
    ```

3. **Configure the Qwak SDK**: Use your account [Qwak API Key](https://docs-saas.qwak.com/docs/getting-started#configuring-qwak-sdk) to set up your SDK locally.

    ```bash
    qwak configure
    ```


4. **Register the Data Source and Feature Set**: Follow the instructions in the [Quickstart Guide](https://docs-saas.qwak.com/docs/feature-store-quickstart-guide) to register the data source and feature set using the code provided in this repository.

    ```bash
    qwak features register
    ```


5. **Run the Model Locally**: Execute the following command to test the model locally:

   ```bash
   python test_model_locally.py
   ```

<br>

Note: Ensure that the data source and feature set have been registered previously as described in the Quickstart Guide.

<br>

## How to Run Remotely on Qwak

1. **Build on the Qwak Platform**:

    Create a new model on Qwak using the command:

    ```bash
    qwak models create "Credit Risk With Feature Store" --project "Sample Project"
    ```


    Initiate a model build with:

    ```bash
    qwak models build --model-id <your-model-id> ./feature_set_quickstart_guide
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
│   ├── data_source.py     # Defines the Data source and Entity
│   ├── feature_set.py     # Defines the Feature set
│   ├── utils.py           # Utilities related to the model
│   ├── model.py           # Defines the Credit Risk Model
│   └── conda.yaml         # Conda environment configurationdata
|
├── tests                  # Empty directory reserved for future test 
│   └── ...                # Future tests
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
