# Streamlit RAG Demo - Retrieval-Augmented Generation with JFrog ML

## Overview

This Streamlit application demonstrates Retrieval-Augmented Generation (RAG) using [JFrog ML's Machine Learning Platform](https://jfrog.com/start-free/). The app combines the power of Large Language Models with vector-based document retrieval to provide contextually relevant answers based on your own knowledge base.

### Features

- **RAG Implementation**: Complete RAG pipeline with vector retrieval and LLM generation
- **Interactive Chat Interface**: User-friendly chat interface for natural conversations
- **Vector Store Integration**: Seamless integration with JFrog ML's Vector Store
- **Contextual Responses**: Answers grounded in your specific document collection
- **Toggle Functionality**: Compare responses with and without vector context
- **LangChain Integration**: Built using LangChain for robust LLM orchestration
- **Real-time Retrieval**: Dynamic document retrieval based on query similarity

### Key Capabilities

- Document-grounded question answering
- Semantic search across document collections
- Context-aware response generation
- Conversation history management
- Configurable retrieval parameters

---

## Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/jfrog/jfrog-ml-examples.git
   cd jfrog-ml-examples/apps/streamlit_rag_demo
   ```

2. **Install Dependencies**
   
   Using Poetry (recommended):
   ```bash
   poetry install
   poetry shell
   ```
   
   Or using pip:
   ```bash
   pip install streamlit streamlit-chat langchain qwak-sdk
   ```

3. **Configure JFrog ML SDK**
   ```bash
   pip install qwak-sdk
   qwak configure
   ```

4. **Set Up Vector Store**
   
   Ensure you have a vector store collection set up in JFrog ML with your documents:
   - Collection name: `financial-data` (or update in `app.py`)
   - Documents should have a `chunk_text` property
   - Vector embeddings should be generated for semantic search

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

   The app will be available at `http://localhost:8501`

---

## Architecture

### RAG Pipeline

The application implements a complete RAG pipeline:

1. **Query Processing**: User input is processed and prepared for retrieval
2. **Vector Retrieval**: Semantic search finds relevant document chunks
3. **Context Preparation**: Retrieved documents are formatted as context
4. **LLM Generation**: Large language model generates response using context
5. **Response Display**: Answer is extracted and displayed to user

### Core Components

```bash
.
├── app.py              # Main Streamlit application
├── chain.py            # LangChain LLM chain configuration
├── vector_store.py     # Vector store retrieval logic
├── qwak_llm.py         # JFrog ML LLM integration
├── pyproject.toml      # Poetry dependencies
├── poetry.lock         # Dependency lock file
├── qwak.png           # Application logo
└── README.md          # Documentation
```

#### `app.py`
- Main Streamlit interface
- Chat history management
- User interaction handling
- Response processing and display

#### `chain.py`
- LangChain configuration
- Prompt template definition
- LLM chain setup with JFrog ML integration

#### `vector_store.py`
- Vector store client integration
- Document retrieval logic
- Context formatting and preparation

#### `qwak_llm.py`
- Custom LangChain LLM wrapper for JFrog ML
- Model inference and response handling

---

## Usage

### Basic RAG Interaction

1. **Start the Application**: Run `streamlit run app.py`
2. **Enable Vector Store**: Check the "Use Vector Store" checkbox to enable RAG
3. **Ask Questions**: Type your question in the input field
4. **View Responses**: The model will provide context-aware answers
5. **Compare Modes**: Toggle the checkbox to compare RAG vs. standard responses

### Vector Store Configuration

The app is configured to use:
- **Collection Name**: `financial-data`
- **Output Property**: `chunk_text`
- **Top Results**: 4 most relevant documents
- **Properties**: Document text content

### Prompt Template

The application uses a carefully crafted prompt template:

```
<<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
Answer the question based on the context below
<</SYS>>

[INST]
{context}
Question: {input}[/INST]
Answer:
```

---

## Configuration

### Model Configuration

Update the model ID in `app.py`:
```python
QWAK_MODEL_ID = 'your-model-id'  # Default: 'llama2'
```

### Vector Store Configuration

Modify vector store settings in `vector_store.py`:
```python
def retrieve_vector_context(
    query: str,
    collection_name: str = "your-collection-name",
    top_results: int = 4,
    properties: List[str] = ["chunk_text"]
):
```

### Retrieval Parameters

Customize retrieval behavior:
- **Top Results**: Number of documents to retrieve (default: 4)
- **Properties**: Document properties to include in context
- **Collection Name**: Vector store collection to search

---

## Vector Store Setup

### Document Ingestion

To set up your vector store:

1. **Prepare Documents**: Format your documents with proper chunking
2. **Generate Embeddings**: Use JFrog ML's embedding models
3. **Create Collection**: Set up a vector store collection
4. **Upload Documents**: Ingest documents with embeddings

### Example Document Structure

```json
{
  "chunk_text": "Your document content here...",
  "metadata": {
    "source": "document_name.pdf",
    "page": 1,
    "section": "introduction"
  }
}
```

### Ingestion Script

The `ingestion/` directory contains scripts for document processing:
- Document chunking
- Embedding generation
- Vector store upload

---

## Advanced Features

### Conversation Memory

The app maintains conversation history:
- **Session State**: Stores chat history in Streamlit session
- **Context Preservation**: Maintains conversation flow
- **History Display**: Shows previous questions and answers

### Context Formatting

Retrieved documents are formatted for optimal LLM consumption:
- **Separator**: Documents separated by `\n\n---\n\n`
- **Relevance Ranking**: Most relevant documents first
- **Content Extraction**: Extracts specific properties from documents

### Response Processing

The app includes response post-processing:
- **Answer Extraction**: Extracts answer from full LLM response
- **Format Cleaning**: Removes system prompts and formatting
- **Error Handling**: Graceful handling of empty or invalid responses

---

## Customization

### Adding New Collections

To add support for multiple document collections:

1. **Update UI**: Add collection selector to Streamlit interface
2. **Modify Retrieval**: Pass collection name to retrieval function
3. **Configure Properties**: Set appropriate properties for each collection

### Custom Prompt Templates

Modify the prompt template in `chain.py`:
```python
prompt_template = """
Your custom prompt template here...
Context: {context}
Question: {input}
Answer:
"""
```

### Enhanced Retrieval

Implement advanced retrieval strategies:
- **Hybrid Search**: Combine semantic and keyword search
- **Reranking**: Re-rank results based on relevance
- **Filtering**: Apply metadata filters to retrieval

---

## Performance Optimization

### Caching

The app uses Streamlit caching:
- **LLM Chain Caching**: `@st.cache_resource` for chain initialization
- **Vector Store Caching**: Cache frequent queries
- **Response Caching**: Cache similar questions

### Retrieval Optimization

- **Top-K Selection**: Optimize number of retrieved documents
- **Embedding Caching**: Cache document embeddings
- **Query Optimization**: Preprocess queries for better retrieval

---

## Troubleshooting

### Common Issues

1. **Vector Store Connection**
   - Verify JFrog ML SDK configuration
   - Check collection name and access permissions
   - Ensure vector store is properly set up

2. **Model Not Responding**
   - Verify model ID is correct and deployed
   - Check JFrog ML account permissions
   - Ensure model is running and accessible

3. **Empty Responses**
   - Check if documents exist in vector store
   - Verify document properties are correctly configured
   - Ensure embeddings are properly generated

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Use Cases

### Document Q&A
- **Technical Documentation**: Answer questions about product manuals
- **Legal Documents**: Query legal contracts and agreements
- **Research Papers**: Search and summarize academic literature

### Knowledge Management
- **Corporate Knowledge**: Access internal company documents
- **Customer Support**: Provide contextual support responses
- **Training Materials**: Interactive learning from training content

### Content Analysis
- **Financial Reports**: Analyze and query financial documents
- **News Articles**: Search and summarize news content
- **Policy Documents**: Query government and policy documents

---

## Try JFrog ML's MLOps Platform for Free

Are you looking to deploy your machine learning models in a production-ready environment within minutes? [JFrog ML](https://jfrog.com/start-free/) offers a seamless platform to build, train, and deploy your models with ease.

Whether you're a data scientist, ML engineer, or developer, JFrog ML provides the tools and support to take your models from development to deployment effortlessly. Explore the platform and start deploying your models today. [Try JFrog ML for free!](https://jfrog.com/start-free/)