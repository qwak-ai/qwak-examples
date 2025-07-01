# Real-time Credit Risk Assessment with Streaming Features and JFrog ML

## Overview

This project demonstrates real-time credit risk assessment using streaming features with [JFrog ML's Machine Learning Platform](https://jfrog.com/start-free/). The model combines batch features from Snowflake with real-time streaming transaction aggregates to provide comprehensive risk assessment, showcasing advanced MLOps techniques including feature store integration, streaming data processing, and real-time inference.

### Features

- **Streaming Feature Integration**: Combines real-time transaction aggregates with batch features
- **Feature Store Integration**: Leverages JFrog ML Feature Store for both batch and streaming features
- **CatBoost Classification**: Uses CatBoost algorithm for robust credit risk prediction
- **Real-time Processing**: Processes streaming transaction data with 1-minute and 1-hour windows
- **Multi-source Data**: Integrates Snowflake batch data with streaming transaction data
- **Automated Feature Extraction**: Automatic feature extraction during inference

### Key Capabilities

- Real-time credit risk scoring
- Streaming transaction analysis with time-based aggregations
- Integration with multiple data sources (Snowflake, streaming data)
- Automated feature engineering and extraction
- Comprehensive risk assessment combining historical and real-time data

---

## Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/jfrog/jfrog-ml-examples.git
   cd jfrog-ml-examples/credit_risk_streaming
   ```

2. **Install Dependencies**
   Make sure you have the required dependencies installed:
   ```bash
   pip install qwak-sdk pandas numpy scikit-learn catboost
   ```

3. **Install and Configure the JFrog ML SDK**
   Use your account [JFrog ML API Key](https://docs.jfrog.com/jfrog-ml/getting-started) to set up your SDK locally:
   ```bash
   pip install qwak-sdk
   qwak configure
   ```

4. **Configure Feature Store Access**
   Ensure your JFrog ML account has access to the required feature sets:
   - `transaction-aggregates-demo` (streaming features)
   - `qwak-snowflake-webinar` (batch features)

---

## Feature Store Integration

This project demonstrates advanced feature store usage with two types of feature sets:

### Streaming Features (`transaction-aggregates-demo`)

Real-time transaction aggregates with different time windows:

#### 1-Minute Window Features
- `count_transaction_amount_1m`: Transaction count in last minute
- `sum_transaction_amount_1m`: Total transaction amount in last minute
- `sample_stdev_transaction_amount_1m`: Standard deviation of amounts
- `median_transaction_amount_1m`: Median transaction amount
- `max_transaction_amount_1m`: Maximum transaction amount

#### 1-Hour Window Features
- `count_transaction_amount_1h`: Transaction count in last hour
- `sum_transaction_amount_1h`: Total transaction amount in last hour
- `sample_stdev_transaction_amount_1h`: Standard deviation of amounts
- `median_transaction_amount_1h`: Median transaction amount
- `max_transaction_amount_1h`: Maximum transaction amount

### Batch Features (`qwak-snowflake-webinar`)

Historical customer data from Snowflake:
- `job`: Customer job category
- `credit_amount`: Credit amount requested
- `duration`: Loan duration
- `purpose`: Loan purpose
- `risk`: Risk label (good/bad) - used as target variable

---

## How to Run Remotely on JFrog ML

1. **Build on the JFrog ML Platform**

   Create a new model on JFrog ML using the command:
   ```bash
   qwak models create "Credit Risk Streaming" --project "Sample Project"
   ```

   Build the model:
   ```bash
   qwak models build \
   --model-id <your-model-id> \
   ./credit_risk_streaming \
   --instance medium
   ```

2. **Deploy the Model on the JFrog ML Platform with a Real-Time Endpoint**

   To deploy your model via the CLI, use the following command:
   ```bash
   qwak models deploy realtime \
   --model-id <your-model-id> \
   --build-id <your-build-id> \
   --instance small \
   --server-workers 1 \
   --feature-extraction-enabled
   ```

   **Note**: The `--feature-extraction-enabled` flag is crucial for automatic feature extraction.

3. **Test the Live Model with a Sample Request**

   The model expects a user ID for feature extraction:
   ```bash
   curl -X POST \
   -H "Authorization: Bearer <your-qwak-token>" \
   -H "Content-Type: application/json" \
   -d '{"user_id": "b0ca3ac4-5432-4c21-8251-a6ae0d3ad874"}' \
   https://models.<your-account>.qwak.ai/<model-id>/v1/predict
   ```

---

## Model Architecture

The model uses **CatBoost**, a gradient boosting algorithm optimized for categorical features:

- **Algorithm**: CatBoost Classifier
- **Training Data**: Combined streaming and batch features
- **Target Variable**: Binary risk classification (good/bad)
- **Feature Engineering**: Automatic handling of categorical features
- **Evaluation Metric**: F1 Score

### Training Configuration

```python
params = {
    'iterations': 100,
    'learning_rate': 0.1,
    'eval_metric': 'Accuracy',
    'logging_level': 'Silent',
    'use_best_model': True
}
```

---

## Input/Output Format

### Input Schema
The model uses feature extraction, so the input is minimal:
- **Field**: `user_id` (string)
- **Description**: Unique identifier for the user/customer
- **Example**: `"b0ca3ac4-5432-4c21-8251-a6ae0d3ad874"`

### Output Format
- **Type**: Pandas DataFrame
- **Columns**:
  - `Risk`: Risk score (0-1, where 1 indicates good risk, 0 indicates bad risk)

### Example Usage

```python
import pandas as pd

# Input data
input_df = pd.DataFrame({
    'user_id': ['b0ca3ac4-5432-4c21-8251-a6ae0d3ad874']
})

# The model will automatically extract features and return:
# Risk: 0.85 (indicating good risk)
```

---

## Feature Extraction Process

The model automatically extracts features during inference through the following process:

1. **Population Data**: Loads population data from `population.csv`
2. **Feature Store Query**: Queries both streaming and batch feature sets
3. **Feature Combination**: Combines features from multiple sources
4. **Automatic Processing**: Handles feature alignment and preprocessing
5. **Prediction**: Generates risk score using the trained CatBoost model

### Feature Extraction Schema

The model schema defines the expected features:

```python
# Streaming features
FeatureStoreInput(entity=user_id, name="transaction-aggregates-demo.count_transaction_amount_1m")
# ... (additional streaming features)

# Batch features  
FeatureStoreInput(entity=user_id, name='qwak-snowflake-webinar.job')
# ... (additional batch features)
```

---

## Project Structure

```bash
.
├── main/                           # Main directory containing core code
│   ├── __init__.py                # Package initialization
│   ├── model.py                   # Credit risk streaming model implementation
│   └── population.csv             # Population data for feature extraction
├── notebook_example/               # Example notebooks
├── Makefile                       # Build automation
└── README.md                      # Documentation
```

---

## Data Sources

### Population Data (`population.csv`)
Contains user identifiers and metadata for feature extraction population definition.

### Streaming Data Source
Real-time transaction data processed through JFrog ML's streaming feature pipeline:
- **Source**: Transaction event stream
- **Processing**: Real-time aggregations with sliding windows
- **Update Frequency**: Real-time (sub-second latency)

### Batch Data Source  
Historical customer data from Snowflake:
- **Source**: Snowflake data warehouse
- **Processing**: Batch feature computation
- **Update Frequency**: Daily/batch updates

---

## Performance Considerations

- **Real-time Inference**: Optimized for low-latency prediction
- **Feature Caching**: Automatic feature caching for improved performance
- **Streaming Processing**: Efficient streaming aggregation computations
- **Categorical Handling**: CatBoost's native categorical feature support
- **Memory Efficiency**: Optimized memory usage for production deployment

---

## Monitoring and Logging

The model includes comprehensive monitoring:

- **Training Metrics**: F1 score, accuracy, learning rate tracking
- **Feature Quality**: Automatic feature validation and monitoring
- **Prediction Logging**: Request/response logging for audit trails
- **Performance Metrics**: Latency and throughput monitoring

### Logged Metrics

During training, the following metrics are logged:
```python
{
    'f1_score': 0.85,
    'iterations': 100,
    'learning_rate': 0.1,
    'accuracy': 0.90,
    'random_state': 42,
    'test_size': 0.25
}
```

---

## Advanced Features

### Multi-Window Analysis
The model analyzes transaction patterns across multiple time windows:
- **Short-term (1 minute)**: Captures immediate transaction behavior
- **Medium-term (1 hour)**: Identifies hourly transaction patterns
- **Long-term (batch)**: Historical customer behavior analysis

### Categorical Feature Handling
Automatic handling of categorical features:
- Job categories
- Loan purposes
- Risk categories
- No manual encoding required

### Feature Store Benefits
- **Consistency**: Same features for training and inference
- **Reusability**: Features shared across multiple models
- **Governance**: Centralized feature management and versioning
- **Performance**: Optimized feature serving infrastructure

---

## Try JFrog ML's MLOps Platform for Free

Are you looking to deploy your machine learning models in a production-ready environment within minutes? [JFrog ML](https://jfrog.com/start-free/) offers a seamless platform to build, train, and deploy your models with ease.

Whether you're a data scientist, ML engineer, or developer, JFrog ML provides the tools and support to take your models from development to deployment effortlessly. Explore the platform and start deploying your models today. [Try JFrog ML for free!](https://jfrog.com/start-free/)