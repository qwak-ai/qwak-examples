
# RAG Based Text Generation Agent with Transformers and JFrogML

## Overview

This project utilizes the Qwen2.5-0.5B-Instruct model for efficient text generation. It's implemented using the [JFrogML's Machine Learning Platform](https://www.qwak.com/) and the Transformers library.

It features:

- **Custom JFrogChatAgent Class Definition**: Customizes the base QwakModel to work with the Qwen2.5 model and augment results from a ChromaDB in-memory Vector Database.
  
- **Model Initialization**: Populates a local ChromaDB instance with text data and initializes the Qwen model with pre-trained weights using Hugging Face's pipeline.

- **Text Generation via JFrogML's Predict API**: Utilizes JFrogML's Predict API for generating text based on input prompts.

The code is designed for seamless integration with JFrogML's platform and serves as a practical example for text generation tasks.



<br>

## How to Test Locally


1. **Clone the Repository**: Clone this GitHub repository to your local machine.

2. **Install Dependencies**: Make sure you have the required dependencies installed, as specified in the `conda.yml` file.

    ```bash
    conda env create -f main/conda.yaml
    conda activate rag-chat-agent
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
    qwak models create "RAG" --project "Sample Project"
    ```


    Initiate a model build with:

    ```bash
    qwak models build --model-id <your-model-id> \
    ./RAG \
    --gpu-compatible \
    --instance medium
    ```


2. **Deploy the Model on the Qwak Platform with a Real-Time Endpoint**:

    To deploy your model via the CLI, use the following command:

    ```bash
    qwak models deploy realtime \
    --model-id <your-model-id> \
    --build-id <your-build-id> \
    --instance gpu.t4.xl \
    --server-workers 1
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
├── main                                    # Main directory containing core code
│   ├── __init__.py                         # An empty file that indicates this directory is a Python package
│   ├── model.py                            # JFrogML model that defines the main functionality of the Chat Agent
│   ├── jfml_local_chat_model.py            # Wraps a locally deployed Qwen model as a ChatBase object
│   ├── lang_graph_chat_agent.py            # Defines the Graph to retrieve and generate
│   ├── chroma_retriever.py                 # Wraps the local Chroma as a Retriever, required for the Lang Graph
│   ├── dummy_data.csv                      # Dummy data for the vector database
│   ├── rag_chain_config.yaml               # Configuration file for all the pieces of the application
│   └── conda.yaml                          # Dependency manifest
|
├── test_model_locally.py  # Script to test the model locally
├── test_live_model.py     # Script to test the live model with a sample REST request
└── README.md              # Documentation
```


<br>
<br>

## Try JFrogML's MLOps Platform for Free

Are you looking to deploy your machine learning models in a production-ready environment within minutes? [JFrogML](https://jfrog.com/start-free/) offers a seamless platform to build, train, and deploy your models with ease.

Whether you're a data scientist, ML engineer, or developer, JFrogML provides the tools and support to take your models from development to deployment effortlessly. Explore the platform and start deploying your models today. [Try JFrogML for free!](https://jfrog.com/start-free/)
