# JFrog ML Examples
Example projects that demonstrate how to build, train, and deploy ML features and models using JFrog ML.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Model Examples](#model-examples)
4. [Feature Store Examples](#feature-store-examples)
5. [Applications](#applications)
6. [Contributing](#contributing)
7. [License](#license)

## Overview

This repository contains example projects that showcase the capabilities of the JFrog ML platform for MLOps. Each project is designed to be a standalone example, demonstrating different aspects of machine learning, from data preprocessing to model building and deployment.

The examples cover various domains including:
- **Natural Language Processing (NLP)**: Text generation, classification, and analysis
- **Computer Vision**: Image classification and analysis
- **Predictive Analytics**: Fraud detection, churn prediction, risk assessment
- **Speech Processing**: Speech-to-text conversion
- **Feature Engineering**: Feature store implementations and streaming features
- **Retrieval-Augmented Generation (RAG)**: Question-answering systems

## Getting Started

To get started with these examples:

1. Clone this repository.
2. Navigate to the example project you're interested in.
3. Follow the README and installation instructions within each project folder.

### Prerequisites
- JFrog ML account and API key
- Python 3.8+ (specific versions may vary by project)
- Conda or Poetry for dependency management (depending on the project)

## Model Examples

### Natural Language Processing (NLP)

| Example | Category | Model | Framework | Info |
|---------|----------|-------|-----------|------|
| [Sentiment Analysis](./bert_conda/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![BERT](https://img.shields.io/badge/-BERT-%23D3D3D3) | ![Conda](https://img.shields.io/badge/-Conda-green) | Binary sentiment analysis using pre-trained BERT model |
| [Basic Text Generation](./bert_conda_generative/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![BERT](https://img.shields.io/badge/-BERT-%23D3D3D3) | ![Conda](https://img.shields.io/badge/-Conda-green) | Text generation using pre-trained BERT model |
| [Text Generation](./flan_t5_poetry/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![FlanT5](https://img.shields.io/badge/-flan--t5--small-%23D3D3D3) | ![Poetry](https://img.shields.io/badge/-Poetry-blue) | T5 model for generic text generation tasks |
| [Financial Text Generation](./flan_t5_finetuned_poetry/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![T5 Base](https://img.shields.io/badge/-t5--base-%23D3D3D3) | ![Poetry](https://img.shields.io/badge/-Poetry-blue) | T5 base model fine-tuned on Financial QA data |
| [Sentiment Classification](./transformers_conda/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![DistilBERT](https://img.shields.io/badge/-DistilBERT-%23D3D3D3) | ![Conda](https://img.shields.io/badge/-Conda-green) | DistilBERT-based text classifier for Yelp reviews |
| [Code Generation](./codegen_conda/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![codegen-350M-mono](https://img.shields.io/badge/codegen--350M--mono-%23D3D3D3) | ![Conda](https://img.shields.io/badge/-Conda-green) | Autoregressive language models for program synthesis |
| [Text Generation (DistilGPT2)](./distilgpt2_conda/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![DistilGPT2](https://img.shields.io/badge/-DistilGPT2-%23D3D3D3) | ![Conda](https://img.shields.io/badge/-Conda-green) | Lightweight GPT-2 model for text generation |
| [Text Generation (GPT-Neo)](./gpt_neo_conda/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![GPT-Neo](https://img.shields.io/badge/-GPT--Neo-%23D3D3D3) | ![Conda](https://img.shields.io/badge/-Conda-green) | Open-source GPT variant for text generation |
| [Text Classification](./text_classification_transformer/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![DistilBERT](https://img.shields.io/badge/-DistilBERT-%23D3D3D3) | ![Transformers](https://img.shields.io/badge/-Transformers-%23D3D3D3) | News classification using DistilBERT with fine-tuning |
| [Movie Review Generation](./transformer_finetuning_jfrog/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![GPT-2](https://img.shields.io/badge/-GPT--2-%23D3D3D3) | ![Transformers](https://img.shields.io/badge/-Transformers-%23D3D3D3) | GPT-2 fine-tuned on IMDb movie reviews |
| [Large Language Model](./llama2/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![Llama-2](https://img.shields.io/badge/-Llama--2-%23D3D3D3) | ![GPU](https://img.shields.io/badge/-GPU-red) | Leading open-source LLM (requires GPU) |

### Computer Vision

| Example | Category | Model | Framework | Info |
|---------|----------|-------|-----------|------|
| [Image Classification](./image_classifier/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![MobileNetV2](https://img.shields.io/badge/-MobileNetV2-%23D3D3D3) | ![PyTorch](https://img.shields.io/badge/-PyTorch-orange) | Image classification using quantized MobileNetV2 |

### Speech Processing

| Example | Category | Model | Framework | Info |
|---------|----------|-------|-----------|------|
| [Speech Recognition](./whisper_speech_recognition/) | ![Speech](https://img.shields.io/badge/-Speech-purple) | ![Whisper](https://img.shields.io/badge/-Whisper-%23D3D3D3) | ![Transformers](https://img.shields.io/badge/-Transformers-%23D3D3D3) | Speech-to-text using OpenAI's Whisper model |

### Predictive Analytics

| Example | Category | Model | Framework | Info |
|---------|----------|-------|-----------|------|
| [Fraud Detection](./fraud_detection/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![XGBoost](https://img.shields.io/badge/-XGBoost-%23D3D3D3) | ![Conda](https://img.shields.io/badge/-Conda-green) | Credit card fraud detection using XGBoost |
| [Credit Risk Assessment](./catboost_poetry/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![CatBoost](https://img.shields.io/badge/-CatBoost-%23D3D3D3) | ![Poetry](https://img.shields.io/badge/-Poetry-blue) | Loan default risk prediction using CatBoost |
| [Customer Churn Analysis](./churn_model_new/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![XGBoost](https://img.shields.io/badge/-XGBoost-%23D3D3D3) | ![Conda](https://img.shields.io/badge/-Conda-green) | Telecom subscriber churn prediction |
| [Titanic Survival Prediction](./titanic_conda/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![CatBoost](https://img.shields.io/badge/-CatBoost-%23D3D3D3) | ![Conda](https://img.shields.io/badge/-Conda-green) | Binary classification for survival prediction |
| [Credit Risk Streaming](./credit_risk_streaming/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![CatBoost](https://img.shields.io/badge/-CatBoost-%23D3D3D3) | ![Streaming](https://img.shields.io/badge/-Streaming-yellow) | Real-time credit risk assessment with streaming features |

### Embeddings & Retrieval

| Example | Category | Model | Framework | Info |
|---------|----------|-------|-----------|------|
| [Sentence Embeddings](./sentence_transformers_poetry/) | ![Embeddings](https://img.shields.io/badge/-Embeddings-orange) | ![all-MiniLM-L12-v2](https://img.shields.io/badge/-all--MiniLM--L12--v2-%23D3D3D3) | ![Poetry](https://img.shields.io/badge/-Poetry-blue) | Semantic search using sentence transformers |
| [Vector Similarity Search](./vector_store/) | ![Embeddings](https://img.shields.io/badge/-Embeddings-orange) | ![VectorStore](https://img.shields.io/badge/-VectorStore-%23D3D3D3) | ![ChromaDB](https://img.shields.io/badge/-ChromaDB-purple) | Product description similarity search |
| [RAG Text Generation](./RAG/) | ![RAG](https://img.shields.io/badge/-RAG-red) | ![Qwen2.5](https://img.shields.io/badge/-Qwen2.5--0.5B-%23D3D3D3) | ![ChromaDB](https://img.shields.io/badge/-ChromaDB-purple) | Retrieval-augmented generation with ChromaDB |
| [QA Bot with Falcon](./qa_bot_falcon_chroma/) | ![RAG](https://img.shields.io/badge/-RAG-red) | ![Falcon](https://img.shields.io/badge/-Falcon-%23D3D3D3) | ![ChromaDB](https://img.shields.io/badge/-ChromaDB-purple) | Question-answering bot using Falcon and ChromaDB |

## Feature Store Examples

| Example | Category | FeatureSet | Info |
|---------|----------|------------|------|
| [Fraud Detection with Feature Store](./feature_set_quickstart_guide/) | ![Guide](https://img.shields.io/badge/-Guide-blue) | ![Batch](https://img.shields.io/badge/-Batch-%23D3D3D3) | Fraud detection with online features |
| [Batch Feature Set with SQL Transformation](./feature_store/batch_feature_set_sql_transformation.ipynb) | ![Guide](https://img.shields.io/badge/-Guide-blue) | ![Batch](https://img.shields.io/badge/-Batch-%23D3D3D3) | Define and use Batch Feature Set with SQL |
| [Batch Feature Set with Koalas Transformation](./feature_store/batch_feature_set_udf_transformation.ipynb) | ![Guide](https://img.shields.io/badge/-Guide-blue) | ![Batch](https://img.shields.io/badge/-Batch-%23D3D3D3) | Batch Feature Set with Koalas (UDF) |
| [Batch Feature Set with Window Aggregations](./feature_store/batch_feature_set_window_agg.ipynb) | ![Guide](https://img.shields.io/badge/-Guide-blue) | ![Batch](https://img.shields.io/badge/-Batch-%23D3D3D3) | SQL Window Aggregations in Feature Sets |

## Applications

| Application | Type | Framework | Description |
|-------------|------|-----------|-------------|
| [Streamlit Demo App](./apps/streamlit_demo_app/) | ![Web App](https://img.shields.io/badge/-Web%20App-lightblue) | ![Streamlit](https://img.shields.io/badge/-Streamlit-red) | Interactive demo application for model testing |
| [Streamlit RAG Demo](./apps/streamlit_rag_demo/) | ![Web App](https://img.shields.io/badge/-Web%20App-lightblue) | ![Streamlit](https://img.shields.io/badge/-Streamlit-red) | RAG-powered question-answering web interface |

## Project Structure

Each project follows a consistent structure:

```
project_name/
├── main/                    # Core model implementation
│   ├── __init__.py         # Package initialization
│   ├── model.py            # Main model class
│   └── conda.yml           # Dependencies (Conda projects)
│   └── pyproject.toml      # Dependencies (Poetry projects)
├── test_model_locally.py   # Local testing script
├── test_live_model.py      # Live model testing script
├── README.md               # Project documentation
└── [additional files]      # Project-specific files
```

## Quick Start Guide

1. **Choose a project** from the examples above
2. **Navigate to the project directory**:
   ```bash
   cd project_name
   ```
3. **Install dependencies**:
   - For Conda projects: `conda env create -f main/conda.yml`
   - For Poetry projects: `poetry install`
4. **Configure JFrog ML SDK**:
   ```bash
   pip install qwak-sdk
   qwak configure
   ```
5. **Test locally**:
   ```bash
   python test_model_locally.py
   ```
6. **Deploy to JFrog ML** (follow project-specific README)

## Contributing

We welcome contributions! Please read our [contributing guidelines](./CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## Try JFrog ML's MLOps Platform for Free

Are you looking to deploy your machine learning models in a production-ready environment within minutes? [JFrog ML](https://jfrog.com/start-free/) offers a seamless platform to build, train, and deploy your models with ease.

Whether you're a data scientist, ML engineer, or developer, JFrog ML provides the tools and support to take your models from development to deployment effortlessly. Explore the platform and start deploying your models today. [Try JFrog ML for free!](https://jfrog.com/start-free/)
