# Qwak Platform Examples

![Qwak Platform](https://github.com/qwak-ai/qwak-examples/raw/main/_static/llm_cover.png)

Example projects that demonstrate how to build, train, and deploy ML features and models using the Qwak platform 🦅.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Projects](#projects)
4. [Contributing](#contributing)
5. [License](#license)

## Overview

This repository contains example projects that showcase the capabilities of the Qwak platform for MLOps. Each project is designed to be a standalone example, demonstrating different aspects of machine learning, from data preprocessing to model building and deployment.

## Getting Started

To get started with these examples:

1. Clone this repository.
2. Navigate to the example project you're interested in.
3. Follow the README and installation instructions within each project folder.


## Qwak Model Examples

| Example | Category | Model | Info |
|---------|------|----------|------|
| [Fraud Detection with Feature Store](./feature_set_quickstart_guide/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![Catboost](https://img.shields.io/badge/-Catboost-%23D3D3D3) ![Feature Store](https://img.shields.io/badge/-Feature%20Store-%23D3D3D3) | Fraud Detection model with inference based on Online Features |
| [Sentiment Analysis](./bert_conda/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![BERT](https://img.shields.io/badge/-BERT-%23D3D3D3) | Performs binary sentiment analysis using a pre-trained BERT model. |
| [Basic Text Generation ](./bert_conda_generative/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![BERT](https://img.shields.io/badge/-BERT-%23D3D3D3) | Generates text using a pre-trained BERT model. |
| [Credit Risk Assesment](./catboost_poetry/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![CatBoost](https://img.shields.io/badge/-CatBoost-%23D3D3D3) | Predicts loan default risk using CatBoost algorithm [Poetry] |
| [Customer Churn Analysis](./churn_model_new/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![XGBoost](https://img.shields.io/badge/-XGBoost-%23D3D3D3) | Predicts Telecom subscriber churn using XGBoost [Conda]. |
| [Code Generation](./codegen_conda/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![codegen-350M-mono](https://img.shields.io/badge/codegen--350M--mono-%23D3D3D3) ![Transformers](https://img.shields.io/badge/-Transformers-%23D3D3D3) | Autoregressive language models for program synthesis and code generation. |
| [Text Generation](flan_t5_poetry/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![FlanT5](https://img.shields.io/badge/-flan--t5--small-%23D3D3D3) ![Transformers](https://img.shields.io/badge/-Transformers-%23D3D3D3) | A small T5 model pre-trained for generic text generation tasks.[Conda] |
| [Financial Text Generation](./flan_t5_finetuned_poetry/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![T5 Base](https://img.shields.io/badge/-t5--base-%23D3D3D3) ![Transformers](https://img.shields.io/badge/-Transformers-%23D3D3D3) | T5 base model trained on Financial QA data for domain specific tasks.[Poetry] |
| [Titanic Survival Prediction](./titanic_conda/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![CatBoost](https://img.shields.io/badge/-CatBoost-%23D3D3D3) | Binary classification model for Titanic survival prediction.[Conda] |
| [Sentiment Classification](./transformers_conda/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![DistilBERT](https://img.shields.io/badge/-DistilBERT-%23D3D3D3) ![Transformers](https://img.shields.io/badge/-Transformers-%23D3D3D3) | DistilBERT-based text classifier for Yelp reviews on Qwak platform.[Conda] |
| [Vector Similarity Search](./vector_store/) | ![Embeddings](https://img.shields.io/badge/-Embeddings-orange) | ![VectorStore](https://img.shields.io/badge/-VectorStore-%23D3D3D3) | Vectorizes product descriptions for similarity-based search. |


## Qwak Feature Store Examples

| Example | Category | FeatureSet | Info |
|---------|------|----------|------|
| [Batch Feature Set with SQL Transformation](./feature_store/batch_feature_set_sql_transformation.ipynb) | ![Guide](https://img.shields.io/badge/-Guide-blue) | ![Batch](https://img.shields.io/badge/-Batch-%23D3D3D3) | Define, register and use a Batch Feature Set in a Qwak Model |


## Contributing

We welcome contributions! Please read our [contributing guidelines](./CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
