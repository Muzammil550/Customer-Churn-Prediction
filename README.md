# **Customer Churn Prediction**
*A real-world machine learning project to predict customer churn using Python and advanced ML models.*

---

## 📌 Project Overview
This project predicts **customer churn** — identifying customers likely to stop using a product or service — by analyzing demographics, service usage, and subscription data.  
The aim is to help businesses **proactively retain customers** by taking targeted action before churn happens.

---

## 📊 Dataset
The dataset includes:
- **Customer demographics** (e.g., age, gender, location)
- **Service usage patterns**
- **Subscription details & contract types**
- **Churn label** indicating whether the customer left or stayed

---

## 🛠 Methodology
1. **Data Cleaning & Preprocessing** – Handling missing values, encoding categorical variables, scaling features  
2. **Exploratory Data Analysis (EDA)** – Identifying churn trends and key patterns  
3. **Feature Engineering** – Creating new variables to improve prediction accuracy  
4. **Model Training** – Compared **Logistic Regression** and **Random Forest** using **GridSearchCV** for hyperparameter tuning  
5. **Evaluation** – Measured performance with Accuracy, Confusion Matrix, and ROC Curve  

---

## 📈 Key Results
| Model | Accuracy |
|-------|----------|
| Logistic Regression | **94.18%** |
| Random Forest | **96.85%** ✅ |

- **Random Forest** performed best, effectively capturing complex relationships in the data  
- Identified top churn factors, such as contract type, tenure, and service usage  

---

## 🛠 Tools & Libraries
- **Python**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Environment**: Anaconda (Spyder / Jupyter Notebook)  

---

## 📂 Project Structure
