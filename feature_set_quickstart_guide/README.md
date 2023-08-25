# Feature Set - Quickstart Guide Project

## Overview

This project demonstrates how to extract, process, store, and consume features using [Qwak's Feature Store](https://www.qwak.com/product/feature-store). It includes examples of defining a Data Source, working with Feature Sets, and running a Credit Risk Machine Learning model. 

The code is designed to work in conjunction with the [Quickstart Guide](https://docs-saas.qwak.com/docs/getting-started-copy) provided by Qwak.

## How to Run Locally

1. **Clone the Repository**: Clone this GitHub repository to your local machine.

2. **Register the Data Source and Feature Set**: Follow the instructions in the [Quickstart Guide](https://docs-saas.qwak.com/docs/feature-store-quickstart-guide) to register the data source and feature set using the code provided in this repository.

    ```bash
    qwak features register
    ```

3. **Install Dependencies**: Make sure you have the required dependencies installed, as specified in the `conda.yml` file.

    ```bash
    conda env create -f main/conda.yaml
    ```


4. **Run the Model Locally**: Execute the following command to test the model locally:

   ```bash
   python test_model_locally.py
   ```

5. **Build on the Qwak Platform**:

    Create a new model on Qwak using the command:

    ```bash
    qwak models create "Credit Risk With Feature Store" --project "Sample Project"
    ```


    Initiate a model build with:

    ```bash
    qwak models build --model-id credit_risk_model ./feature_set_quickstart_guide
    ```

6. **Deploy the Model on the Qwak Platform with a Real-Time Endpoint**:

    To deploy your model via the CLI, use the following command:

    ```bash
    qwak models deploy realtime --model-id credit_risk_model --build-id <your-build-id>
    ```



Note: Ensure that the data source and feature set have been registered previously as described in the Quickstart Guide.

## Project Structure

```bash
.
├── main                   # Main directory containing core code
│   ├── __init__.py        # An empty file that indicates this directory is a Python package
│   ├── model.py           # Defines the Credit Risk Model
│   ├── feature_set.py     # Defines the feature set
│   ├── data_source.py     # Defines the CSV data source for credit risk 
│   └── conda.yaml         # Conda environment configurationdata
|
├── tests                  # Empty directory reserved for future test 
│   └── ...                # Future tests
|
├── test_model_locally.py  # Script to test the model locallyimplementations
└── README.md              # This file
```


<br>
<br>

## Try Qwak's MLOps Platform for Free

Are you looking to deploy your machine learning models in a production-ready environment within minutes? [Qwak](https://www.qwak.com/) offers a seamless platform to build, train, and deploy your models with ease.

Whether you're a data scientist, ML engineer, or developer, Qwak provides the tools and support to take your models from development to deployment effortlessly. Explore the platform and start deploying your models today. [Try Qwak for free!](https://www.qwak.com/)
