# 🚀 Remote Training & Deployment (JFrogML)

## 🎯 Overview

This guide covers how to build and deploy the customer churn prediction model on JFrogML. The **training happens during the build process** via the `build()` method in your FrogMLModel, and the `predict()` method handles serving.

---

## 🔐 Login & Configure

- Install CLI:
```bash
pip install frogml-cli
```

- Login and configure credentials (interactive):
```bash
frogml config add --interactive
```
Refer to the JFrog ML install and setup instructions: [Install JFrog ML](https://jfrog.com/help/r/jfrog-ml-documentation/install-jfrog-ml).

---

## 📁 JFrogML Project Structure

Understanding the required project structure for JFrogML deployment:

```
.
├── main/                       # Main directory containing core code
│   ├── __init__.py             # Python package marker (required)
│   ├── model.py                # FrogMLModel implementation with build() and predict()
│   ├── data.csv                # Customer churn training dataset
│   └── conda.yml               # Conda environment dependencies
```

### **File Explanations**

- **`main/`**: Directory containing all core model code and dependencies
- **`__init__.py`**: Empty file that makes `main/` a Python package for imports
- **`model.py`**: FrogMLModel class with key methods:
  - `build()`: Training logic (runs during build process)
  - `initialize_model()`: Runtime initialization at deployment
  - `predict()`: Inference logic (runs during serving)
  - `schema()`: Input/output validation
- **`data.csv`**: Customer churn training dataset used by the `build()` method
- **`conda.yml`**: Environment dependencies (Python version, packages, etc.)

### **Workflow Integration**
1. **Build Process**: JFrogML reads `conda.yml` → creates environment → imports from `main/model.py` → runs `build()` method → packages everything
2. **Deployment**: Uses the trained model and `predict()` method for serving

---

## 🧪 Local Testing

Before building on JFrogML, validate your code locally for faster feedback:

```bash
# Test your model locally using JFrogML's run_local utility
python test_model_code_locally.py
```

This uses JFrogML's `run_local` SDK utility to:
- Validate your `FrogMLModel` implementation
- Test `build()` and `predict()` methods locally
- Catch issues before triggering remote builds
- Provide faster development iteration

---

## 🎯 Create Model in JFrog UI

Before building, create your model in the JFrog platform:

1. **Navigate to JFrog UI** → **AI/ML** section
2. **Create New Model** → Name: "Churn Prediction Model" 
3. **Copy the Model ID** generated (you'll need this for CLI commands)

This associates your code with a specific model in the JFrog platform for tracking and management.

---

## 🏗️ Build (Training + Packaging)

The build process executes your `build()` method (which contains training logic) and packages everything for deployment:

```bash
# Build the model (run from churn_model_new/ directory - the . picks up code from current dir)
frogml models build --model-id churn_prediction_model . --instance medium

# This will return a Build ID (UUID) - copy it for deployment
# Example output: Build ID: f47ac10b-58cc-4372-a567-0e02b2c3d479

# View build logs (includes training logs)
frogml models builds logs -b <your_build_id> -f 

# See all build command parameters
frogml models build --help
```

**What happens during build:**
1. **Training**: Your `build()` method runs with XGBoost hyperparameter optimization
2. **Packaging**: Creates deployment-ready container with trained model
3. **Validation**: Ensures model and serving logic are ready
4. **Build ID Generated**: Copy this ID for deployment commands

---

## 🚀 Deploy

### Real-time API
```bash
# Deploy as real-time endpoint (use the Build ID from previous step)
frogml models deploy realtime --model-id churn_prediction_model --build-id <your-build-id>

# See all realtime deployment parameters
frogml models deploy realtime --help
```

Test the endpoint:
```bash
python test_live_endpoint.py
```

### Batch Processing
```bash
# Deploy for batch inference (use the Build ID from previous step)
frogml models deploy batch --model-id churn_prediction_model --build-id <your-build-id>

# See all batch deployment parameters
frogml models deploy batch --help
```


Submit a batch job (example):
```bash
python test_batch_endpoint.py
```


---

## 🎯 JFrogML Platform Benefits

### **Integrated ML Lifecycle**
- **Code to Production**: Single platform for building, training, and serving ML models
- **FrogMLModel Framework**: Standardized approach with `build()` for training and `predict()` for serving
- **Scalable Infrastructure**: Auto-scaling real-time and batch inference endpoints

### **Enterprise-Grade Features**
- **JFrog Integration**: Seamless integration with JFrog Artifactory for model artifacts
- **Security & Governance**: Enterprise security controls and model governance
- **Monitoring & Observability**: Built-in model performance monitoring and logging

### **Developer Experience**
- **CLI & UI**: Flexible interaction via command line or web interface  
- **Model Versioning**: Automatic versioning and artifact management
- **Testing Tools**: Local testing capabilities before deployment
