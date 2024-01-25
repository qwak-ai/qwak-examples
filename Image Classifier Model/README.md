Image Classifier Model with ResNet and Qwak
Overview
This project utilizes a specialized version of the ResNet model, known as ResNet50, for classifaction of images. It's implemented using the Qwak's Machine Learning Platform.

It features:

Custom ImageClassifier Class Definition: Customizes the base QwakModel to work with the ResNet50 model.

Model Initialization: Initializes the ResNet50 model with pre-trained weights.

Text Generation via Qwak's Predict API: Utilizes Qwak's Predict API for generating text based on input prompts.

The code is designed for seamless integration with Qwak's platform and serves as a practical example for text generation tasks.


How to Test Locally
Clone the Repository: Clone this GitHub repository to your local machine.


pip install qwak-sdk
qwak configure
Run the Model Locally: Execute the following command to test the model locally:

poetry run python test_model_locally.py


How to Run Remotely on Qwak
Build on the Qwak Platform:

Create a new model on Qwak using the command:

qwak models create "Image classifier" --project "Image Classification"
Initiate a model build with:

qwak models build --model-id <your-model-id> ./Image Classifier Model
Deploy the Model on the Qwak Platform with a Real-Time Endpoint:

To deploy your model via the CLI, use the following command:

qwak models deploy realtime --model-id <your-model-id> --build-id <your-build-id>
Test the Live Model with a Sample Request:

Install the Qwak Inference SDK:

pip install qwak-inference
Call the Real-Time endpoint using your Model ID from the Qwak platform:

python test_live_mode.py <your-qwak-model-id>

Project Structure
.
├── main                   # Main directory containing core code
│   ├── __init__.py        # An empty file that indicates this directory is a Python package
│   ├── model.py           # Defines the Code Generation Model
│   └── conda.yaml        # Conda environment configuration data
|
├── test_model_locally.py  # Script to test the model locally
├── test_live_model.py     # Script to test the live model with a sample REST request
└── README.md              # Documentation


Try Qwak's MLOps Platform for Free
Are you looking to deploy your machine learning models in a production-ready environment within minutes? Qwak offers a seamless platform to build, train, and deploy your models with ease.

Whether you're a data scientist, ML engineer, or developer, Qwak provides the tools and support to take your models from development to deployment effortlessly. Explore the platform and start deploying your models today. Try Qwak for free!
