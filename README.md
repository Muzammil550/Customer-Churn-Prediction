# 📊 Customer Churn Prediction

## 📌 Project Overview
Customer churn is when a customer stops using a company's service.  
Predicting churn is critical for **customer retention strategies** because retaining existing customers is often more cost-effective than acquiring new ones.

This project builds a **machine learning pipeline** to predict whether a customer will churn based on transaction history, engagement data, and behavioral metrics.

---

## 🎯 Objectives
- **Identify** customers likely to churn.
- **Analyze** which features most influence churn.
- **Compare** multiple ML models for prediction performance.
- **Provide** insights to improve retention strategies.

---

## 🗂 Dataset
Example of the dataset structure:

| custid | retained | created    | firstorder | lastorder  | esent | eopenrate | eclickrate | avgorder | ordfreq | paperless | refill | doorstep | favday | city |
|--------|----------|------------|------------|------------|-------|-----------|------------|----------|---------|-----------|--------|----------|--------|------|
| 6H6T6N | 0        | 9/28/2012  | 11/08/2013 | 11/08/2013 | 29    | 100       | 3.45       | 14.52    | 0       | 0         | 0      | 0        | Monday | CHO  |
| APCENR | 1        | 12/19/2010 | 01/04/2011 | 19/01/2014 | 95    | 92.63     | 10.53      | 83.69    | 0.18    | 1         | 1      | 1        | Friday | CHO  |

**Key Columns:**
- `retained`: Target variable (1 = retained, 0 = churned)
- `esent`, `eopenrate`, `eclickrate`: Email engagement metrics
- `avgorder`, `ordfreq`: Purchase patterns
- `favday`: Favorite day for purchases
- `city`: Customer location

---

## 🛠 Tools & Technologies
- **Python**: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
- **ML Models**: Logistic Regression, Random Forest, Neural Network, Gradient Boosting, XGBoost, LightGBM, CatBoost
- **Feature Selection**: Random Forest importance-based selection
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- **PDF Reporting**: FPDF

---

## 🔍 Approach
1. **Data Preprocessing**
   - Date parsing and missing value handling
   - Feature engineering: recency, tenure, order intervals, ratios
   - One-hot encoding for categorical variables (`favday`, `city`)

2. **Feature Selection**
   - Random Forest feature importance
   - Keep features with importance ≥ 0.005

3. **Model Training**
   - Hyperparameter tuning with `GridSearchCV`
   - Models tested: Logistic Regression, Random Forest, Neural Network, Gradient Boosting, XGBoost, LightGBM, CatBoost

4. **Evaluation**
   - Compare Accuracy and F1 Score
   - Generate confusion matrices & ROC curves
   - Save all results in a **PDF report** for business presentation

---

## 📈 Results

**Final Model Accuracies & F1 Scores:**

| Model              | Accuracy | F1 Score |
|--------------------|----------|----------|
| Random Forest      | 0.9685   | 0.9804   |
| XGBoost            | 0.9685   | 0.9803   |
| Neural Network     | 0.9674   | 0.9796   |
| Gradient Boosting  | 0.9664   | 0.9791   |
| Logistic Regression| 0.9418   | 0.9637   |

---

## 📊 Feature Importance (Top 10)

![Feature Importance](visualizations/top_features.png)

---

## 📉 Model Performance Charts

**Accuracy Comparison:**
![Accuracy Comparison](visualizations/model_accuracy.png)

**F1 Score Comparison:**
![F1 Score Comparison](visualizations/model_f1_score.png)

---

## 📄 PDF Report
The project automatically generates a PDF file:  
**`Churn_Model_Report.pdf`**  
This report contains:
- Selected features list
- Feature importance plot
- Confusion matrices for all models
- ROC curves for all models
- Accuracy and F1 Score comparison charts

---

## ▶️ How to Run
1. **Clone the repository**
```bash
git clone https://github.com/Muzammil550/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
