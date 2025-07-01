# Streamlit Demo App for JFrog ML Models

## Overview

This Streamlit application provides an interactive web interface for testing and demonstrating various Large Language Models (LLMs) deployed on [JFrog ML's Machine Learning Platform](https://jfrog.com/start-free/). The app features a chat-like interface where users can interact with different models and view conversation embeddings in real-time.

### Features

- **Interactive Chat Interface**: Chat-like UI for natural conversation with LLMs
- **Multiple Model Support**: Easy switching between different deployed models
- **Real-time Embeddings**: Automatic generation and visualization of conversation embeddings
- **Model Comparison**: Side-by-side comparison of different model responses
- **Token Management**: Automatic authentication and token management
- **Responsive Design**: Modern, user-friendly interface built with Streamlit

### Supported Models

The app currently supports the following JFrog ML deployed models:

- **FLAN T5**: Generic text generation model (`flan_t5`)
- **Fine-tuned T5**: Domain-specific fine-tuned model (`fine_tuned_flan_t5`)
- **Sentence Embeddings**: Semantic embeddings model (`sentence_embeddings`)
- **Falcon 7B**: Large language model for conversation (`falcon_7b`)
- **Pythia (PEFT)**: Parameter-efficient fine-tuned model (`parameter_efficient_fine_tuning`)

---

## Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/jfrog/jfrog-ml-examples.git
   cd jfrog-ml-examples/apps/streamlit_demo_app
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - `streamlit`
   - `streamlit-chat`
   - `streamlit-extras`
   - `requests`
   - `Pillow`
   - `qwak-inference`

3. **Configure API Key**
   
   Update the `API_KEY` variable in `app.py` with your JFrog ML API key:
   ```python
   API_KEY = 'your-actual-api-key-here'
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

   The app will be available at `http://localhost:8501`

---

## Usage

### Basic Interaction

1. **Select a Model**: Use the dropdown to choose which model to interact with
2. **Enter Your Question**: Type your question or prompt in the text input field
3. **View Response**: The model's response will appear in the chat interface
4. **Explore Embeddings**: Expand the embeddings section to view conversation vectors

### Model Selection

The app provides easy switching between different models:

- **FLAN T5**: Best for general question-answering and text generation
- **Fine-tuned T5**: Optimized for specific domain tasks (e.g., financial QA)

### Embeddings Visualization

For each conversation, the app automatically:
1. Combines the question and answer into a single text
2. Generates semantic embeddings using the sentence transformer model
3. Displays the embeddings in an expandable section

---

## Architecture

### Application Structure

```bash
.
├── app.py                 # Main Streamlit application
├── inference.py           # Model inference and API communication
├── model_testing.py       # Local model testing utilities
├── requirements.txt       # Python dependencies
├── llm_cover.png         # Application header image
└── README.md             # Documentation
```

### Core Components

#### `app.py`
- Main Streamlit interface
- Session state management
- UI layout and components
- Model selection and interaction flow

#### `inference.py`
- JFrog ML API communication
- Token management and authentication
- Model-specific inference functions
- Embeddings generation

### Authentication Flow

1. **API Key Configuration**: Set your JFrog ML API key
2. **Token Generation**: Automatic token generation on app startup
3. **Session Management**: Token stored in Streamlit session state
4. **API Calls**: Authenticated requests to JFrog ML endpoints

---

## API Integration

### JFrog ML REST API

The app uses JFrog ML's REST API for model inference:

```python
# Authentication
POST https://grpc.qwak.ai/api/v1/authentication/qwak-api-key

# Model Inference
POST https://models.llm-demo.qwak.ai/v1/{model_id}/predict
```

### Model-Specific Endpoints

Each model has its own inference format:

#### Text Generation Models
```python
{
    'columns': ["prompt"],
    'index': [0],
    "data": [["question: " + user_input]]
}
```

#### Embeddings Model
```python
{
    'columns': ["text"],
    'index': [0],
    "data": [["Question: " + input_text]]
}
```

---

## Configuration

### Environment Variables

You can configure the app using environment variables:

- `QWAK_API_KEY`: Your JFrog ML API key
- `QWAK_ACCOUNT`: Your JFrog ML account name
- `MODEL_ENDPOINT`: Custom model endpoint (optional)

### Model Configuration

To add new models, update the `MODELS` dictionary in `app.py`:

```python
MODELS = {
    "Your Model Name": {
        "model_id": "your_model_id",
        "fn": partial(get_api_inference, qwak_token=st.session_state['qwak_token'])
    }
}
```

---

## Features in Detail

### Chat Interface

- **Message History**: Maintains conversation context
- **User/Bot Distinction**: Clear visual separation of messages
- **Real-time Responses**: Streaming-like response display
- **Loading Indicators**: Visual feedback during model inference

### Model Switching

- **Dropdown Selection**: Easy model switching without page reload
- **Session Persistence**: Selected model remembered during session
- **Dynamic Configuration**: Models loaded dynamically from configuration

### Embeddings Visualization

- **Automatic Generation**: Embeddings created for each conversation
- **Expandable Display**: Collapsible embeddings section
- **Vector Inspection**: Full embedding vector display
- **Context Preservation**: Shows the combined question-answer text

---

## Customization

### Adding New Models

1. **Deploy Model on JFrog ML**: Ensure your model is deployed and accessible
2. **Update Model List**: Add model configuration to `MODELS` dictionary
3. **Custom Inference Function**: Create model-specific inference function if needed
4. **Test Integration**: Verify the model works through the interface

### UI Customization

The app uses Streamlit components that can be customized:

- **Layout**: Modify column layouts and containers
- **Styling**: Add custom CSS through `st.markdown`
- **Components**: Add new Streamlit components for enhanced functionality
- **Branding**: Replace the header image with your own branding

### Response Processing

Customize how model responses are processed:

```python
def custom_response_processor(response):
    # Add custom processing logic
    return processed_response
```

---

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify your API key is correct
   - Check if your JFrog ML account has model access
   - Ensure token is being generated successfully

2. **Model Not Responding**
   - Verify model is deployed and running
   - Check model ID matches the deployed model
   - Confirm model endpoint is accessible

3. **Embeddings Not Generating**
   - Ensure sentence embeddings model is deployed
   - Check if embeddings endpoint is responding
   - Verify input format is correct

### Debug Mode

Enable debug logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Performance Considerations

- **Token Caching**: Tokens are cached in session state to avoid repeated authentication
- **Async Processing**: Consider implementing async requests for better performance
- **Response Caching**: Cache model responses to improve user experience
- **Connection Pooling**: Use connection pooling for multiple API requests

---

## Security Considerations

- **API Key Protection**: Never commit API keys to version control
- **Environment Variables**: Use environment variables for sensitive configuration
- **Token Expiration**: Handle token expiration gracefully
- **Input Validation**: Validate user inputs before sending to models

---

## Try JFrog ML's MLOps Platform for Free

Are you looking to deploy your machine learning models in a production-ready environment within minutes? [JFrog ML](https://jfrog.com/start-free/) offers a seamless platform to build, train, and deploy your models with ease.

Whether you're a data scientist, ML engineer, or developer, JFrog ML provides the tools and support to take your models from development to deployment effortlessly. Explore the platform and start deploying your models today. [Try JFrog ML for free!](https://jfrog.com/start-free/)