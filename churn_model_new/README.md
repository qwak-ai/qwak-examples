# 📞 Customer Churn Prediction Model

## 🎯 Overview

This project demonstrates a **Customer Churn Prediction Model** using XGBoost and the JFrogML platform. It showcases multiple deployment strategies and training approaches for production-ready churn prediction systems.

## 📋 Prerequisites

Before starting, ensure you have:

- **Python 3.9-3.11** installed
- **JFrog account** ([Get started for free](https://jfrog.com/start-free/))

<br>

## 🚀 Quick Start

### ☁️ **ML App Code → Build → Deploy**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   💻 Local ML   │ -> │   🏗️ Build      │ -> │   🚀 Deploy     │
│   App Code      │    │    Process      │    │   ML Serving    │
│   (or GitHub)   │    │(w/ Training Job)│    │   API Endpoint  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
     Local IDE              FrogML CLI           FrogML CLI
```

**Complete workflow**: [🚀 Remote Training & Deployment Guide](remote-training-and-deployment.md)

**Best for**: Standardized, replicable, production-ready workflows with integrated training and serving

<br>


## 📁 Project Structure

```
churn_model_new/
├── main/                       # Main directory containing core code
│   ├── __init__.py             # Python package initialization
│   ├── model.py                # FrogMLModel with churn prediction logic
│   ├── data.csv                # Training dataset
│   └── conda.yml               # Environment dependencies
├── tests/                      # Integration tests
│   └── it/
│       └── test_churn.py       # Integration test for churn model
├── test_model_code_locally.py  # Script to test the model locally
├── test_live_endpoint.py       # Script to test live deployment endpoint
├── test_batch_endpoint.py      # Script to test batch inference endpoint
├── README.md                   # This documentation
└── remote-training-and-deployment.md  # Deployment guide
```

<br>

## 🔗 Related Resources

- [JFrogML Documentation](https://jfrog.com/help/r/jfrog-ml-documentation/jfrog-ml-introduction)
- [Feature Store Example](../feature_set_quickstart_guide/README.md)


## 📚 Next Steps

1. **Choose your deployment path** from the guides above
2. **Follow the step-by-step instructions** in your chosen guide
3. **Customize the model** for your specific churn prediction needs
4. **Scale up** with larger datasets and more complex feature engineering

---

**Ready to get started?** Pick a guide above and begin your churn prediction journey! 🚀