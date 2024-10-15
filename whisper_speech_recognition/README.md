# Whisper Speech Recognition Model with Transformers and JFrogML

## Overview

This project demonstrates the integration of a pre-trained Whisper model for speech recognition with [JFrogML's Machine Learning Platform](https://www.qwak.com/). The key components include:

- Whisper-based speech-to-text conversion
- Using JFrogML's API for inference
- Local and remote testing of the model

---

## Setup and Installation

1. **Clone the Repository**
    To get started, clone this repository to your local machine:

    ```bash
    git clone https://github.com/qwak-ai/qwak-examples.git
    ```

2. **Install Dependencies**: Install the required dependencies using the `conda.yml` file.

    ```bash
    conda env create -f main/conda.yaml
    conda activate whisper_speech_recognition
    ```

3. **Install and Configure the Qwak SDK**: Use your account [Qwak API Key](https://docs.qwak.com/docs/getting-started#configuring-qwak-sdk) to set up your SDK locally.

    ```bash
    pip install qwak-sdk
    qwak configure
    ```

4. **Install FFmpeg**: Install [FFmpeg](https://ffmpeg.org/) based on your platform (MacOS, Windows, or Linux).

5. **Run the Model Locally**: Execute the following command to test the model locally:

   ```bash
   python test_model_locally.py
   ```


<br>

<br>

## How to Run Remotely on JFrogML

1. **Build on the JFrogML Platform**:

    Create a new model on Qwak using the command:

    ```bash
    qwak models create "Whisper ASR" --project "Sample Project"
    ```


    Build the model:

    ```bash
    qwak models build \
    --model-id <your-model-id> \
    ./whisper_speech_recognition \
    --gpu-compatible \
    --no-validate-serving-artifact \
    -E WHISPER_MODEL=large-v2
    ```


2. **Deploy the Model on the JFrogML Platform with a Real-Time Endpoint**:

    To deploy your model via the CLI, use the following command:

    ```bash
    qwak models deploy realtime \
    --model-id <your-model-id> \
    --build-id <your-build-id> \
    --instance 'gpu.t4.xl' \
    --server-workers 1 \
    --timeout 60000
    ```


## Testing the Live Model

You can test your deployed model on JFrogML using either the Python script or the provided shell script.

### 1. Using Python

This Python script sends an audio file to the model's prediction API for inference.

#### Prerequisites
- Ensure you have Python 3.9+ installed.
- Install the `requests` library:

    ```bash
    pip install requests
    ```

- Set the QWAK_TOKEN environment variable with your Qwak token:

    ```bash
    export QWAK_TOKEN=<your_qwak_token>
    ```

#### Usage

Run the Python script by providing your JFrogML account name and model ID:

```bash
python test_live_model.py <account_name> <model_id> 
```

<br>

- `<account_name>`: Your JfrogML account name.
- `<model_id>`: The ID of the model you want to test, you can find it on your JFrogML model page.

<br>

**Example**:
```bash
export QWAK_TOKEN=your_qwak_token_here
python test_live_model.py acme-organisation whisper_asr
```
<br>

### 2. Using the Shell Script

This script uses `curl` to send an audio file to your deployed model via a REST API.

#### Prerequisites
- Ensure `curl` is installed.
- Install the `requests` library:

    ```bash
    pip install requests
    ```

- Set the QWAK_TOKEN environment variable with your Qwak token:

    ```bash
    export QWAK_TOKEN=<your_qwak_token>
    ```

#### Usage

Run the Python script by providing your JFrogML account name and model ID:

```bash
./test_live_model.sh <account_name> <model_id> 
```

<br>

- `<account_name>`: Your JfrogML account name.
- `<model_id>`: The ID of the model you want to test, you can find it on your JFrogML model page.

<br>

**Example**:
```bash
export QWAK_TOKEN=your_qwak_token_here
./test_live_model.sh acme-organisation whisper_asr
```

<br>


## Project Structure

```bash
.
├── main                   # Main directory containing core code
│   ├── __init__.py        # An empty file that indicates this directory is a Python package
│   ├── model.py           # Defines the Code Generation Model
│   └── conda.yaml         # Conda environment configurationdata
│
├── test_model_locally.py  # Script to test the model locally
├── test_live_model.sh     # Shell script to test the live model with a sample REST request
├── test_live_model.py     # Python script to test the live model with a sample REST request
├── harvard.wav            # Sample audio for testing purposes
└── README.md              # Documentation
```


<br>
<br>

## Try JFrogML's MLOps Platform for Free

Are you looking to deploy your machine learning models in a production-ready environment within minutes? [JFrogML](https://www.qwak.com/) offers a seamless platform to build, train, and deploy your models with ease.

Whether you're a data scientist, ML engineer, or developer, Qwak provides the tools and support to take your models from development to deployment effortlessly. Explore the platform and start deploying your models today. [Try JFrogML for free!](https://www.qwak.com/)
