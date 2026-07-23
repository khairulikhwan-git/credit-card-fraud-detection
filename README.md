# 💳 Credit Card Fraud Detection & Risk Optimization

## 📌 Executive Summary
This project builds an end-to-end Machine Learning pipeline to detect fraudulent credit card transactions in an extremely imbalanced dataset (0.17% fraud rate across ~285,000 transactions). 

Rather than focusing solely on raw accuracy—which is misleading in imbalanced contexts—this project evaluates models on **Precision vs. Recall trade-offs** and optimizes decision thresholds to align with real-world banking operations (balancing fraud losses against customer friction).

---

## 🔑 Key Results & Model Comparison

By transitioning from a baseline Logistic Regression model to a tuned Random Forest Classifier, **false alarms (False Positives) were reduced by 99%** while maintaining an **86% fraud detection rate**.

| Model Strategy | Decision Threshold | Caught Fraud (TP) | Missed Fraud (FN) | False Alarms (FP) | Precision | Recall | Business Alignment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** *(Balanced)* | 0.50 | 89 | 9 | 1,494 | 6% | 91% | **High Friction:** Catches fraud, but declines 1,400+ legitimate transactions. |
| **Random Forest** *(Default)* | 0.50 | 79 | 19 | 5 | 94% | 81% | **Conservative:** Near-zero false alarms, but lets 19 frauds slip through. |
| **Random Forest** *(Optimized)* | **0.30** | **84** | **14** | **14** | **86%** | **86%** | **Optimal Balance:** Captures high fraud volume while preserving smooth customer UX. |

---

## 🛠️ Project Architecture & Pipeline

```

Raw Data (284,807 rows) 
   │
   ├──► Feature Engineering: Extracted cyclic 'Hour' feature from raw 'Time' elapsed
   │
   ├──► Preprocessing: Scaled 'Amount' & 'Hour' via StandardScaler
   │
   ├──► Stratified Data Split: 80% Train / 20% Test (Preserved 0.17% fraud ratio)
   │
   ├──► Model Benchmarking: Logistic Regression vs. Random Forest
   │
   └──► Decision Optimization: Probability Threshold Tuning (0.50 ➔ 0.30)
```

---

## 💡 Technical & Business Insights

### 1. The Accuracy Trap & Class Imbalance
With 99.83% of transactions being legitimate, a dummy model predicting "Normal" for every transaction would achieve 99.83% accuracy while missing 100% of fraud. Models were instead evaluated using **Precision, Recall, F1-Score, and Confusion Matrices**.

### 2. Threshold Optimization (The "Sweet Spot")
Standard classification models apply a hard `0.50` probability threshold. By evaluating model probabilities (`predict_proba`) across thresholds from 0.10 to 0.50, tuning the threshold to **0.30** yielded the optimal operational sweet spot—increasing Recall from 81% to 86% with minimal addition to false alarms.

### 3. Real-World Risk Deployment Strategy
In modern financial architecture, predictions from this model map directly into a **Multi-Tiered Action Framework**:
* **High Confidence (> 85% Fraud Probability):** Automated transaction block.
* **Medium Confidence (20% – 85% Fraud Probability):** Dynamic Friction (triggers 2FA / SMS OTP / Face ID verification to clear false positives seamlessly).
* **Low Confidence (< 20% Fraud Probability):** Instant auto-approval.

---

## 🧰 Tech Stack
* **Language:** Python 3.x
* **Data Processing & EDA:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (`LogisticRegression`, `RandomForestClassifier`, `StandardScaler`, `train_test_split`)
* **Metrics & Evaluation:** `classification_report`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`
* **Visualization:** Matplotlib, Seaborn

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/khairulikhwan-git/credit-card-fraud-detection.git
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Run the Jupyter / Kaggle Notebook:
   ```bash
   jupyter notebook fraud_detection.ipynb
   ```
