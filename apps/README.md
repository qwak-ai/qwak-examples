# JFrog ML Applications

This directory contains interactive applications built on top of JFrog ML models, demonstrating practical use cases and providing user-friendly interfaces for model interaction.

## Available Applications

### 1. Streamlit Demo App
**Location**: `./streamlit_demo_app/`

An interactive web application for testing and demonstrating various Large Language Models deployed on JFrog ML.

**Features:**
- Chat-like interface for natural conversation with LLMs
- Multiple model support with easy switching
- Real-time embeddings generation and visualization
- Token management and authentication
- Model comparison capabilities

**Supported Models:**
- FLAN T5 (Generic text generation)
- Fine-tuned T5 (Domain-specific tasks)
- Sentence Embeddings (Semantic search)
- Falcon 7B (Conversational AI)
- Pythia (Parameter-efficient fine-tuning)

**Quick Start:**
```bash
cd streamlit_demo_app
pip install -r requirements.txt
streamlit run app.py
```

---

### 2. Streamlit RAG Demo
**Location**: `./streamlit_rag_demo/`

A Retrieval-Augmented Generation (RAG) application that combines Large Language Models with vector-based document retrieval for contextually relevant answers.

**Features:**
- Complete RAG pipeline implementation
- Vector store integration with JFrog ML
- Interactive chat interface
- Context-aware response generation
- Toggle between RAG and standard responses
- LangChain integration for robust orchestration

**Use Cases:**
- Document Q&A systems
- Knowledge management
- Customer support with contextual responses
- Research paper analysis
- Corporate knowledge access

**Quick Start:**
```bash
cd streamlit_rag_demo
poetry install
streamlit run app.py
```

---

## Common Prerequisites

All applications require:

1. **JFrog ML Account**: Active JFrog ML account with API access
2. **Python 3.8+**: Compatible Python version
3. **JFrog ML SDK**: Configured SDK for model access
   ```bash
   pip install qwak-sdk
   qwak configure
   ```

## Architecture Overview

These applications demonstrate different interaction patterns with JFrog ML:

### Direct Model Inference
The Streamlit Demo App shows direct model inference patterns:
- REST API integration
- Token-based authentication
- Multiple model management
- Response formatting and display

### RAG Pattern
The RAG Demo demonstrates advanced ML patterns:
- Vector store integration
- Document retrieval and ranking
- Context injection into prompts
- LangChain orchestration
- Conversation memory management

## Development Guidelines

### Adding New Applications

To add a new application to this directory:

1. **Create Application Directory**
   ```bash
   mkdir your_app_name
   cd your_app_name
   ```

2. **Application Structure**
   ```
   your_app_name/
   ├── app.py              # Main application file
   ├── requirements.txt    # Dependencies
   ├── README.md          # Application documentation
   └── [additional files] # App-specific files
   ```

3. **Documentation Requirements**
   - Clear setup instructions
   - Usage examples
   - Configuration options
   - Troubleshooting guide

4. **Update Main README**
   Add your application to this README with:
   - Brief description
   - Key features
   - Quick start instructions
   - Use cases

### Best Practices

#### Code Organization
- **Modular Design**: Separate concerns into different modules
- **Configuration Management**: Use environment variables or config files
- **Error Handling**: Implement comprehensive error handling
- **Logging**: Add appropriate logging for debugging

#### User Experience
- **Intuitive Interface**: Design user-friendly interfaces
- **Loading Indicators**: Show progress during model inference
- **Error Messages**: Provide clear error messages to users
- **Responsive Design**: Ensure applications work on different screen sizes

#### Security
- **API Key Management**: Never hardcode API keys
- **Input Validation**: Validate all user inputs
- **Authentication**: Implement proper authentication flows
- **Rate Limiting**: Consider rate limiting for production use

## Deployment Options

### Local Development
All applications can be run locally for development and testing:
```bash
streamlit run app.py
```

### Production Deployment
For production deployment, consider:

1. **Containerization**
   ```dockerfile
   FROM python:3.9-slim
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py"]
   ```

2. **Cloud Deployment**
   - Streamlit Cloud
   - Heroku
   - AWS/GCP/Azure
   - Docker containers

3. **Environment Configuration**
   - Set production environment variables
   - Configure logging levels
   - Set up monitoring and alerting

## Common Issues and Solutions

### Authentication Problems
- **Issue**: API key not working
- **Solution**: Verify API key is correct and has proper permissions

### Model Access Issues
- **Issue**: Model not responding
- **Solution**: Check if model is deployed and accessible

### Performance Issues
- **Issue**: Slow response times
- **Solution**: Implement caching, optimize queries, consider async processing

### Dependency Conflicts
- **Issue**: Package version conflicts
- **Solution**: Use virtual environments, pin dependency versions

## Contributing

When contributing to applications:

1. **Follow Standards**: Adhere to existing code standards
2. **Test Thoroughly**: Test all functionality before submitting
3. **Document Changes**: Update documentation for any changes
4. **Consider Security**: Review security implications of changes

## Support and Resources

- **JFrog ML Documentation**: [https://docs.jfrog.com/jfrog-ml](https://docs.jfrog.com/jfrog-ml)
- **Streamlit Documentation**: [https://docs.streamlit.io](https://docs.streamlit.io)
- **LangChain Documentation**: [https://docs.langchain.com](https://docs.langchain.com)

## Try JFrog ML's MLOps Platform for Free

Ready to build your own ML applications? [JFrog ML](https://jfrog.com/start-free/) provides the infrastructure and tools you need to deploy and manage machine learning models at scale. 

Start building today with our comprehensive MLOps platform. [Try JFrog ML for free!](https://jfrog.com/start-free/)