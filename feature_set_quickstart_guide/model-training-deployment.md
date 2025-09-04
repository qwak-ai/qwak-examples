# 🚀 Model Training & Deployment (with Feature Store)

## 🎯 Overview

This guide covers how to build and deploy the credit risk model with Feature Store integration on JFrogML. The **training happens during the build process** via the `build()` method, and the `predict()` method handles serving with real-time Feature Store access.

**Prerequisites**: Complete [Feature Store Setup & Testing](feature-store-setup.md) first.

---

## 🔐 Prerequisites Check

✅ **Feature Store Setup Complete**: You should have already completed the [Feature Store Setup & Testing](feature-store-setup.md) guide, which includes:
- JFrogML CLI installation and configuration
- Data source and feature set registration
- Feature Store validation

If you haven't completed Phase 1, please do that first before proceeding.

---

## 📁 JFrogML Project Structure

Understanding the required project structure for JFrogML deployment:

```
.
├── main/                       # Main directory containing core code
│   ├── __init__.py             # Python package marker (required)
│   ├── model.py                # FrogMLModel implementation with build() and predict()
│   ├── utils.py                # Data preprocessing utilities
│   └── conda.yaml              # Conda environment dependencies
```

### **File Explanations**

- **`main/`**: Directory containing all core model code and dependencies
- **`__init__.py`**: Empty file that makes `main/` a Python package for imports
- **`model.py`**: FrogMLModel class with key methods:
  - `build()`: Training logic (runs during build process)
  - `initialize_model()`: Runtime initialization at deployment
  - `predict()`: Inference logic (runs during serving) with Feature Store integration
  - `schema()`: Defines which features to pull from Feature Store for enrichment
- **`utils.py`**: Data preprocessing and cleaning utilities
- **`conda.yaml`**: Environment dependencies (Python version, packages, etc.)

### **Workflow Integration**
1. **Build Process**: JFrogML reads `conda.yaml` → creates environment → imports from `main/model.py` → runs `build()` method → packages everything
2. **Deployment**: Uses the trained model and `predict()` method for serving with Feature Store access

---
<br>

## 🧪 Step 1: Local Testing

Before building on JFrogML, validate your code locally for faster feedback:

```bash
# Test your model locally using JFrogML's run_local utility
python test_model_locally.py
```

This uses JFrogML's `run_local` SDK utility to:
- Validate your `FrogMLModel` implementation
- Test `build()` and `predict()` methods locally with Feature Store
- Catch issues before triggering remote builds
- Provide faster development iteration

---
<br>

## 🎯 Step 2: Create Model in JFrog UI

Before building, create your model in the JFrog platform:

1. **Navigate to JFrog UI** → **AI/ML** section
2. **Create New Model** → Name: "Credit Risk with Feature Store" 
3. **Copy the Model ID** generated (you'll need this for CLI commands)

This associates your code with a specific model in the JFrog platform for tracking and management.

---
<br>

## 🏗️ Step 3: Build (Training + Packaging)

The build process executes your `build()` method (which contains training logic) and packages everything for deployment:

```bash
# Build the model (run from feature_set_quickstart_guide/ directory - the . picks up code from current dir)
frogml models build --model-id credit_risk_model . --instance medium

# This will return a Build ID (UUID) - copy it for deployment
# Example output: Build ID: f47ac10b-58cc-4372-a567-0e02b2c3d479

# View build logs (includes training logs)
frogml models builds logs -b <your_build_id> -f 

# See all build command parameters
frogml models build --help
```

**What happens during build:**
1. **Feature Store Connection**: Connects to registered Feature Store components
2. **Training**: Your `build()` method runs with offline features from Feature Store
3. **Packaging**: Creates deployment-ready container with trained model and Feature Store connections
4. **Validation**: Ensures model and serving logic are ready with Feature Store integration
5. **Build ID Generated**: Copy this ID for deployment commands

---

## 🚀 Step 4: Deploy

### Real-time API
```bash
# Deploy as real-time endpoint (use the Build ID from previous step)
frogml models deploy realtime --model-id credit_risk_model --build-id <your-build-id>

# See all realtime deployment parameters
frogml models deploy realtime --help
```

Test the endpoint:
```bash
python test_live_model.py
```

<br>

## 🎯 JFrogML Platform Benefits

### **Integrated ML Lifecycle with Feature Store**
- **Code to Production**: Single platform for building, training, and serving ML models with Feature Store
- **FrogMLModel Framework**: Standardized approach with `build()` for training and `predict()` for serving
- **Feature Store Integration**: Seamless online/offline feature access during training and inference
- **Scalable Infrastructure**: Auto-scaling real-time and batch inference endpoints with feature serving

### **Enterprise-Grade Features**
- **JFrog Integration**: Seamless integration with JFrog Artifactory for model artifacts
- **Security & Governance**: Enterprise security controls and model governance
- **Feature Store Management**: Centralized feature engineering, versioning, and serving
- **Monitoring & Observability**: Built-in model performance monitoring and feature drift detection

### **Developer Experience**
- **CLI & UI**: Flexible interaction via command line or web interface  
- **Model Versioning**: Automatic versioning and artifact management
- **Feature Store Tools**: Easy feature registration, backfill, and monitoring
- **Testing Tools**: Local testing capabilities with Feature Store integration before deployment
