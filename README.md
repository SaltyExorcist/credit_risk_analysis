﻿# 📘 Credit Risk Prediction Project Report

## 🧠 Problem Statement  
**Goal**: Predict whether a loan applicant poses a *good* or *bad* credit risk using machine learning.  
**Why it matters**: Financial institutions rely on accurate credit scoring to minimize defaults and maximize lending efficiency.

---

## 📊 Dataset Overview  
We used the **German Credit dataset** containing:
- 1000 rows, ~10 key features (e.g., age, employment, savings, credit amount)
- Target: **Risk** (good / bad)

**Preprocessing Steps**:
- Removed index column (`Unnamed: 0`)
- Handled missing values in `"Saving accounts"` and `"Checking account"` by imputing `'unknown'`
- Applied **Label Encoding** to categorical variables and **Standard Scaling** to numerical features

---

## ⚙️ Feature Engineering  
We simulated the `Risk` label (for model training) based on:
- High `Credit amount` + Short `Duration` → **bad risk**
- Others → **good risk**

This proxy target was used to create a binary classification model.

---

## 🤖 Model Training  
- Used **Random Forest Classifier** with default parameters (optionally tuned via `GridSearchCV`)
- Split data 80/20 for train/test
- Evaluation:
  - Accuracy, F1-score, confusion matrix, SHAP explanations
- Saved model: `credit_risk_model.pkl`

✅ Also stored the `feature_names_in_` to ensure Streamlit input matches model expectations.

---

## 📈 Performance Report
Example results (may vary slightly with retraining):

```
Classification Report:
              precision    recall  f1-score   support
    good         1.00       1.00      1.00       180
    bad          1.00       1.00      1.00        20

Confusion Matrix:
[[180  0]
 [ 0  20]]
```

→ Model performs well on majority class (`good`), and moderately well on minority (`bad`).

---

## 📌 Interpretability with SHAP  
We used **TreeExplainer** to compute SHAP values:

### Global Importance  
Top features influencing predictions:
- `Credit amount` (positively correlated with bad risk)
- `Duration` (shorter duration → more risky)
- `Saving accounts`, `Job`, `Age`

### Local Explanation (Streamlit)  
Users see:
- Predicted risk class (`good` / `bad`)
- SHAP force plot or bar chart showing which features influenced that result most

---

## 💡 Streamlit App Demo  
Built with:
- Form inputs for applicant features
- Real-time model prediction
- SHAP visualization of top contributors

> Command to run:
```bash
 python -m streamlit run app.py
```

---

## 🧪 Exploratory Visualizations  
Included:
- Countplot of `Risk` distribution to check class imbalance
- Pairwise SHAP feature impact (optional force/bar charts)

---


## 📁 Files Included
- `data_preprocessing.py`: feature encoding & scaling
- `train_model.py`: training & evaluation
- `interpret_model.py`: SHAP analysis
- `app.py`: Streamlit front-end
- `credit_risk_analysis.ipynb`: step-by-step notebook
