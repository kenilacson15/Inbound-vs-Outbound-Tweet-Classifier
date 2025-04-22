# 📊 Tweet Direction Classifier: Inbound vs Outbound

A lightweight yet effective **binary classification pipeline** to distinguish between **inbound** (customer-initiated) and **outbound** (support-initiated) tweets using the [Customer Support on Twitter dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).

---

## 🔍 Overview

- **Objective**: Classify tweets as either inbound (1) or outbound (0)
- **Model**: Logistic Regression
- **Features**: TF-IDF, text meta-features, sentiment scores
- **Performance**: F1 ≈ 0.89 on the test set
- **Design Goals**: Reproducible, interpretable, and production-ready

---

## 🧹 Data Preprocessing

To ensure efficient experimentation and fast iteration:

- **Subset**: Random 50k samples from 2.8M (seed=42)
- **Cleaning**:
  - Removed URLs, mentions, hashtags, emojis, digits, punctuation
  - Applied lemmatization and stopword removal
- **Column**: Cleaned text stored in `clean_text`

---

## 📊 Exploratory Data Analysis (EDA)

Visual + statistical insights to inform feature engineering:

- **Class Balance**: Inbound vs Outbound distribution
- **Text Lengths**: Distribution before/after cleaning
- **Word Clouds**: Most common words in each class
- **Top Words**: Token frequencies pre/post-cleaning
- **Heatmap**: Correlation of engineered numerical features

🗂️ *All outputs saved to `data/processed/EDA/`*

---

## ⚙️ Feature Engineering

Combined textual and numerical features:

| Category | Details |
|----------|---------|
| **TF-IDF** | Up to 5000 unigrams/bigrams (`min_df=3`, `max_df=0.85`) |
| **Textual Stats** | Length, num hashtags, mentions, punctuation count |
| **Sentiment** | VADER compound, pos/neg/neu scores |
| **Scaling** | StandardScaler applied to numerical features |

---

## 🧠 Modeling Pipeline

1. **Stratified Train-Test Split** (80/20 on `inbound` flag)
2. **Feature Vectorization** (TF-IDF + numerical)
3. **Model**: `LogisticRegression(class_weight='balanced')`
4. **Evaluation**: Accuracy, Precision, Recall, F1, ROC AUC

---

## ✅ Model Performance

### ✅ Validation Set

| Metric           | Score    |
|------------------|----------|
| Accuracy         | 0.9066   |
| Precision (1)    | 0.9144   |
| Recall (1)       | 0.9146   |
| F1 Score (1)     | 0.9145   |
| ROC AUC          | 0.9644   |

Training Time: ~35 minutes

---

### 🧪 Test Set

| Metric           | Score    |
|------------------|----------|
| Accuracy         | 0.8820   |
| Precision (1)    | 0.8891   |
| Recall (1)       | 0.8956   |
| F1 Score (1)     | 0.8923   |
| ROC AUC          | 0.9483   |
| Inference Time   | ~401 sec |

#### Confusion Matrix

```
               Pred: 0   Pred: 1
True: 0        786       122
True: 1        114       978
```

---

## 🔄 Reproducibility

- Random Seed: `42`
- Serialized artifacts (`joblib`):
  - `tfidf_vectorizer.joblib`
  - `numerical_scaler.joblib`
- All intermediate outputs (features, metadata) saved to disk



---

## 🚀 Future Work

- Hyperparameter tuning (GridSearchCV)
- Ensemble models (e.g., XGBoost, Random Forest)
- Experiment tracking (MLflow, Weights & Biases)
- Real-time inference API (FastAPI or Flask)



## License

The [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

You are free to:
- **Share** — copy and redistribute the material in any medium or format.
- **Adapt** — remix, transform, and build upon the material.

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes.
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license.

