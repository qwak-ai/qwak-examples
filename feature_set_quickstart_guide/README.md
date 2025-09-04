# 🏪 Credit Risk Model with Feature Store

This project demonstrates a **Credit Risk Assessment Model** using JFrogML's Feature Store - from feature engineering to model deployment with real-time inference.

## 📋 Prerequisites

- **Python 3.9-3.11** installed
- **JFrog account** ([Get started for free](https://jfrog.com/start-free/))

<br>

## 🚀 JFrogML Feature Store Workflow

**Sequential workflow** - Phase 2 depends on Phase 1 completion:

### 🏪 **Phase 1: Feature Store Setup** *(Required First)*

```
┌─────────────────┐    ┌─────────────────┐    
│   📊 Data       │ -> │   🔧 Feature    │ 
│   Source        │    │   Set           │
│   Registration  │    │   Registration  │
└─────────────────┘    └─────────────────┘
```

**Complete workflow**: [🏪 Feature Store Setup & Testing Guide](feature-store-setup.md)

**Purpose**: Set up data connectors, feature transformations, and validate Feature Store components

<br>

---

### 🚀 **Phase 2: Model Training & Deployment** *(Depends on Phase 1)*

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   💻 ML App     │ -> │   🏗️ Build      │ -> │   🚀 Deploy     │
│   Code          │    │   (Training)    │    │   ML Serving    │
│   + Features    │    │   Offline Store │    │   Online Store  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Complete workflow**: [🚀 Model Training & Deployment Guide](model-training-deployment.md)

**Purpose**: Build and deploy ML models using the registered Feature Store components

<br>

---


## 📁 Project Structure

```
feature_set_quickstart_guide/
|
├── README.md                    # This overview guide
├── feature_store/               # Feature Store components
│   ├── data_source.py           # Data connector (S3 CSV)
│   └── feature_set.py           # Feature transformations
|
├── main/                        # ML model code
│   ├── __init__.py              # Python package initialization
│   ├── model.py                 # CatBoost credit risk model
│   ├── utils.py                 # Data utilities
│   └── conda.yaml               # Environment dependencies
|
├── feature-store-setup.md       # 🏪 Phase 1 guide
└── model-training-deployment.md # 🚀 Phase 2 guide
```
