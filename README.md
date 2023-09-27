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


## Projects

| Example | Category | Model | Info |
|---------|------|----------|------|
| [Fraud Detection with Feature Store](./feature_set_quickstart_guide/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![Catboost](https://img.shields.io/badge/-Catboost-%23D3D3D3) ![Feature Store](https://img.shields.io/badge/-Feature%20Store-%23D3D3D3) | Fraud Detection model with inference based on Online Features |
| [Text Generation](./science_bot_falcon_chroma/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![falcon-7b](https://img.shields.io/badge/-falcon--7b-%23D3D3D3) ![Transformers](https://img.shields.io/badge/-Transformers-%23D3D3D3) ![Torch](https://img.shields.io/badge/-Torch-%23D3D3D3)| Generates text based on prompts using Falcon-7B model. |
| [Sentiment Analysis](./bert_conda/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![BERT](https://img.shields.io/badge/-BERT-%23D3D3D3) | Performs sentiment analysis using BERT. |
| [Basic Text Generation ](./bert_conda_generative/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![BERT](https://img.shields.io/badge/-BERT-%23D3D3D3) | Generates text using a pre-trained BERT model. |
| [Credit Risk Assesment](./catboost_poetry/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![CatBoost](https://img.shields.io/badge/-CatBoost-%23D3D3D3) | Predicts loan default risk using CatBoost algorithm [Poetry] |
| [Customer Churn Analysis](./churn_model_new/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![XGBoost](https://img.shields.io/badge/-XGBoost-%23D3D3D3) | Predicts Telecom subscriber churn using XGBoost [Conda]. |
| [Code Generation](./codegen_conda/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![codegen-350M-mono](https://img.shields.io/badge/codegen--350M--mono-%23D3D3D3) ![Transformers](https://img.shields.io/badge/-Transformers-%23D3D3D3) | Generates code snippets based on text prompts. |
| [Text Generation](./distilgpt2_conda/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![distilGPT2](https://img.shields.io/badge/-distilGPT2-%23D3D3D3) ![Transformers](https://img.shields.io/badge/-Transformers-%23D3D3D3) | Simplified GPT-2 model for faster, resource-efficient text generation. (GPU Compatible) |
| [Poetry](./falcon_poetry/) | ![App](https://img.shields.io/badge/-App-yellow) | ![N/A](https://img.shields.io/badge/-N%2FA-%23D3D3D3) | Poetry |
| [Text Generation](./flan_t5_conda/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![FlanT5](https://img.shields.io/badge/-FlanT5-%23D3D3D3) ![Transformers](https://img.shields.io/badge/-Transformers-%23D3D3D3) | Text Generation |
| [Poetry - T5](./flan_t5_finetuned_poetry/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![T5](https://img.shields.io/badge/-T5-%23D3D3D3) | Poetry |
| [Text Generation](./gpt_neo_conda/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![gpt-neo-125M](https://img.shields.io/badge/-gpt--neo--125M-%23D3D3D3) ![Transformers](https://img.shields.io/badge/-Transformers-%23D3D3D3) | This model generates text based on a given prompt using GPT-Neo.[Conda] |
| [Financial Text Generation](./pythia_peft_pip/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![pythia-410m-deduped](https://img.shields.io/badge/-pythia--410m--deduped-%23D3D3D3) | Text generation model fine-tuned for financial data.[Pip] |
| [Text/Sentence Embedding](./sentence_transformers_poetry/) | ![Embedding Model](https://img.shields.io/badge/-Embedding--Model-orange) | ![N/A](https://img.shields.io/badge/-all--MiniLM--L12--v2-%23D3D3D3) | Converts text to numerical embeddings.[Poetry] |
| [Titanic Survival Prediction](./titanic_conda/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![CatBoost](https://img.shields.io/badge/-CatBoost-%23D3D3D3) | Binary classification model for Titanic survival prediction.[Conda] |
| [Sentiment Classification](./transformers_conda/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![DistilBERT](https://img.shields.io/badge/-DistilBERT-%23D3D3D3) ![Transformers](https://img.shields.io/badge/-Transformers-%23D3D3D3) | DistilBERT-based text classifier for Yelp reviews on Qwak platform.[Conda] |
| [Vector Similarity Search](./vector_store/) | ![Generative](https://img.shields.io/badge/-Generative-green) | ![VectorStore](https://img.shields.io/badge/-VectorStore-%23D3D3D3) | Vectorizes product descriptions for similarity-based search. |
| [Churn Prediction](./xgboost_conda/) | ![Predictive](https://img.shields.io/badge/-Predictive-blue) | ![XGBoost](https://img.shields.io/badge/-XGBoost-%23D3D3D3) | Predicts customer churn using XGBoost and telecom data.[Conda] |



## Contributing

We welcome contributions! Please read our [contributing guidelines](./CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
