## 📌 Overview
This project predicts whether a customer will churn based on transaction history, engagement data, and behavioral patterns.  
It uses feature engineering, categorical encoding, and Random Forest feature selection to identify the most important predictors, and compares multiple ML models.

---

## 🗂 Dataset
Example:
| custid | retained | created    | firstorder | lastorder  | esent | eopenrate | eclickrate | avgorder | ordfreq | favday | city |
|--------|----------|------------|------------|------------|-------|-----------|------------|----------|---------|--------|------|
| 6H6T6N | 0        | 9/28/2012  | 11/08/2013 | 11/08/2013 | 29    | 100       | 3.45       | 14.52    | 0       | Monday | CHO  |
| APCENR | 1        | 12/19/2010 | 01/04/2011 | 19/01/2014 | 95    | 92.63     | 10.53      | 83.69    | 0.18    | Friday | CHO  |

---

## 🛠 Tools & Libraries
- Python, Pandas, NumPy, Scikit-learn
- Seaborn, Matplotlib
- Category Encoders
- XGBoost, LightGBM, CatBoost
- FPDF (for PDF report generation)

---

## 🔍 Approach
1. **Data Preprocessing** — Handle dates, missing values, feature engineering (recency, tenure, ratios)
2. **Feature Selection** — Random Forest importance ≥ 0.005
3. **Model Training** — Logistic Regression, Random Forest, Neural Network, Gradient Boosting, XGBoost, LightGBM, CatBoost
4. **Evaluation** — Accuracy, F1 Score, Confusion Matrix, ROC Curve

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

## 📊 Visualizations
### Feature Importance
![Feature Importance](visualizations/<img width="83%" alt="image" src="https://github.com/user-attachments/assets/6e82cf82-9c97-494d-95c7-299d1d514b6d" />
)

### Confusion Matrix with Accuracy and F1 Score 
<img width="83%" alt="Screenshot 2025-09-06 110116" src="https://github.com/user-attachments/assets/598e71d8-1722-4e29-b7ba-d0a8071d30fd" />

### ROC Curve
<img width="83%" alt="Screenshot 2025-09-06 110137" src="https://github.com/user-attachments/assets/0be44e0a-de75-4f94-ab75-5be927970f8c" />

---

## 📄 PDF Report
The project generates a **[Churn_Model_Report.pdf](Churn_Model_Report.pdf)** with:
- Selected features
- Feature importance chart
- Confusion matrices
- ROC curves
- Accuracy & F1 comparison

---

## ▶️ How to Run
```bash
git clone https://github.com/Muzammil550/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
python churn_prediction.py
