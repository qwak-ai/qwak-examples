# 🏪 Feature Store Setup & Testing

## 🎯 Overview

This guide covers how to set up and test JFrogML's Feature Store components. You'll register data sources, create feature sets, and validate the feature pipeline before building ML models.

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

## 📁 Feature Store Project Structure

Understanding the Feature Store components:

```
.
├── feature_store/
│   ├── data_source.py          # Data connector definition
│   └── feature_set.py          # Feature transformations and scheduling
├── main/
│   └── utils.py                # Data preprocessing utilities
```

### **File Explanations**

- **`feature_store/data_source.py`**: Defines connector to raw data (CSV from S3)
- **`feature_store/feature_set.py`**: Transforms raw data into features with Spark SQL, scheduling, and storage
- **`main/utils.py`**: Data cleaning and preprocessing utilities

---

## 🗃️ Step 1: Data Source Registration

### **Validate Data Source Connection**
Before registration, test your data source locally:

```python
# In a Python cell or script
from feature_store.data_source import csv_source

# Test data source connectivity and sample data
sample_data = csv_source.get_sample()
print(sample_data.head())
```

This validates:
- S3 connectivity and access
- Data format and structure
- Column names and data types

### **Register Data Source**
Once validated, register the connector to your raw data:

```bash
# Register data source (run from feature_set_quickstart_guide/ directory)
frogml features register -p feature_store/data_source.py
```

**What this does:**
- Creates connection to S3 CSV data
- Defines data access configuration
- Makes raw data available to Feature Store

**Data Source Configuration:**
```python
csv_source = CsvSource(
    name='credit_risk_data',
    path='s3://qwak-public/example_data/data_credit_risk.csv',
    date_created_column='date_created',
    filesystem_configuration=AnonymousS3Configuration(),
)
```

---

## 🔧 Step 2: Feature Set Registration

### **Validate Feature Transformations**
Before registration, test your feature transformation logic locally:

```python
# In a Python cell or script
from feature_store.feature_set import user_features

# Test feature transformation logic
transformed_sample = user_features.get_sample()
print(transformed_sample.head())
```

This validates:
- SQL transformation logic
- Feature engineering correctness
- Output schema and data types

### **Register Feature Set**
Once validated, transform raw data into features and set up offline/online storage:

```bash
# Register feature set (data transformation + storage)
frogml features register -p feature_store/feature_set.py
```

**What this does:**
- Applies Spark SQL transformations to raw data
- Creates **Offline Store** (historical features for training)
- Creates **Online Store** (real-time features for inference)
- Sets up daily scheduling at midnight
- Backfills historical data from 2015

**Feature Set Configuration:**
```python
@batch.feature_set(name="user-credit-risk-features", key="user_id")
@batch.scheduling(cron_expression="0 0 * * *")  # Daily updates
@batch.backfill(start_date=datetime(2015, 1, 1))
def user_features():
    return SparkSqlTransformation("""
        SELECT user_id as user,
               age, job, credit_amount, duration,
               housing, saving_account, checking_account,
               purpose, sex, date_created
        FROM credit_risk_data
    """)
```

---

## 🔍 Feature Store Architecture

### **Runtime Flow**
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Source   │    │   Feature Set    │    │  Feature Store  │
│   (CSV from S3) │───▶│   (Transform)    │───▶│  Serving Runtime│
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                               ┌──────────────────┬──────────────────┐
                               │  Offline Store   │  Online Store    │
                               │  (Historical)    │  (Real-time)     │
                               └──────────────────┴──────────────────┘
```

### **Key Concepts**

**Configuration Layer:**
- **Entity**: `user` - unique identifier for feature vectors
- **Data Source**: Connector definition to raw data (S3 CSV)
- **Feature Set**: Transformation logic + scheduling configuration
- **Scheduling**: Automatic feature updates (daily at midnight)
- **Backfill**: Historical data processing (2015 to present)

**Storage Layer (Actual Manifestations):**
- **Offline Store**: Physical storage of historical features for model training
- **Online Store**: Physical storage of real-time features for model inference

---

## 🛠️ Troubleshooting

**Feature Set Issues**: Navigate to **JFrogML UI** → **AI/ML** → **Feature Sets** → `user-credit-risk-features` → **Executions** → **Logs**

**Local Validation Issues**: Re-run the validation steps within each registration phase to identify data source connectivity or transformation problems.


---

## 🎯 Next Steps

**Proceed to Model Integration**: [🚀 Model Training & Deployment Guide](model-training-deployment.md)

Your Feature Store is now ready to serve features to ML models. The next guide shows how to build and deploy models that consume features from both the offline store (for training) and online store (for real-time inference).
