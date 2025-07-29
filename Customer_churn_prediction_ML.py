import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, RocCurveDisplay

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ✅ Create visualizations folder
os.makedirs("visualizations", exist_ok=True)

# -----------------------------
# Load Data
# -----------------------------
train_df = pd.read_excel(r"C:\Users\WAJAHAT TRADERS\Downloads\RR-Train-forSPSS (1) (2).xlsx", sheet_name='relay train')
test_df = pd.read_excel(r"C:\Users\WAJAHAT TRADERS\Downloads\RR-Test-forSPSS (1) (1).xlsx", sheet_name='relay test.csv')

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess(df):
    for col in ["firstorder", "lastorder", "created"]:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    df['days_since_firstorder'] = (pd.Timestamp.now() - df['firstorder']).dt.days
    df['days_since_lastorder'] = (pd.Timestamp.now() - df['lastorder']).dt.days
    df['days_since_created'] = (pd.Timestamp.now() - df['created']).dt.days
    df['days_between_orders'] = (df['lastorder'] - df['firstorder']).dt.days
    df['avg_days_per_order'] = df['days_between_orders'] / df['ordfreq']
    df['avg_days_per_order'].replace([np.inf, -np.inf], 0, inplace=True)
    df['avg_days_per_order'].fillna(0, inplace=True)
    df['tenure'] = df['days_since_firstorder']
    df['recency_ratio'] = df['days_since_lastorder'] / df['tenure']
    df['recency_ratio'].replace([np.inf, -np.inf], 0, inplace=True)
    df['recency_ratio'].fillna(0, inplace=True)
    df['days_since_firstorder_created'] = df['days_since_created'] - df['days_since_firstorder']
    return df

train_df = preprocess(train_df)
test_df = preprocess(test_df)

# -----------------------------
# Features & Target
# -----------------------------
X_train = train_df.drop(['custid','retained','created','firstorder','lastorder','train'], axis=1)
X_test = test_df.drop(['custid','retained','created','firstorder','lastorder','train'], axis=1)
y_train = train_df['retained']
y_test = test_df['retained']

# -----------------------------
# Encode Categorical Variables
# -----------------------------
encoder = ce.OneHotEncoder(cols=['favday','city'], use_cat_names=True)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# -----------------------------
# Feature Selection (Random Forest)
# -----------------------------
rf_selector = RandomForestClassifier(n_estimators=200, random_state=42)
rf_selector.fit(X_train, y_train)

importances = pd.Series(rf_selector.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.head(15), y=importances.head(15).index, palette="viridis")
plt.title("Top 15 Features by Random Forest Importance")
plt.savefig("visualizations/top_features.png")
plt.close()

threshold = 0.005
selected_features = importances[importances >= threshold].index.tolist()
X_train = X_train[selected_features]
X_test = X_test[selected_features]

# -----------------------------
# Scale Continuous Features
# -----------------------------
continuous_cols = [col for col in selected_features if X_train[col].dtype != 'uint8']
scaler = StandardScaler()
X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

# -----------------------------
# Train & Evaluate
# -----------------------------
accuracies = {}
f1_scores = {}

def train_and_evaluate(name, model, params):
    grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracies[name] = acc
    f1_scores[name] = f1

    print(f"\n{name}")
    print("Best Params:", grid.best_params_)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(f"visualizations/confusion_matrix_{name.replace(' ', '_')}.png")
    plt.close()

    RocCurveDisplay.from_estimator(grid.best_estimator_, X_test, y_test)
    plt.title(f"{name} - ROC Curve")
    plt.savefig(f"visualizations/roc_curve_{name.replace(' ', '_')}.png")
    plt.close()

# -----------------------------
# Run Models
# -----------------------------
train_and_evaluate("Logistic Regression", LogisticRegression(max_iter=1000),
                   {'C':[0.1, 1, 10], 'penalty':['l1','l2'], 'solver':['liblinear','saga']})

train_and_evaluate("Random Forest", RandomForestClassifier(random_state=42),
                   {'n_estimators':[100,200], 'max_depth':[10,20,None], 'min_samples_split':[2,5]})

train_and_evaluate("Neural Network", MLPClassifier(max_iter=500, random_state=42),
                   {'hidden_layer_sizes':[(50,),(100,)], 'activation':['relu','tanh'], 'alpha':[0.0001,0.001]})

train_and_evaluate("Gradient Boosting", GradientBoostingClassifier(random_state=42),
                   {'n_estimators':[100,200], 'learning_rate':[0.05,0.1], 'max_depth':[3,5]})

train_and_evaluate("XGBoost", XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False),
                   {'n_estimators':[100,200], 'learning_rate':[0.05,0.1], 'max_depth':[3,5]})

train_and_evaluate("LightGBM", LGBMClassifier(random_state=42),
                   {'n_estimators':[100,200], 'learning_rate':[0.05,0.1], 'max_depth':[-1,10]})

train_and_evaluate("CatBoost", CatBoostClassifier(verbose=0, random_state=42),
                   {'iterations':[100,200], 'learning_rate':[0.05,0.1], 'depth':[4,6]})

# -----------------------------
# Plot Results (Sorted) 
# -----------------------------
sorted_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
sorted_names = [name for name, _ in sorted_models]
sorted_acc = [accuracies[name] for name in sorted_names]
sorted_f1 = [f1_scores[name] for name in sorted_names]

plt.figure(figsize=(10,6))
sns.barplot(x=sorted_names, y=sorted_acc, palette="mako")
plt.xticks(rotation=30)
plt.title("Model Accuracy Comparison (Sorted)")
plt.savefig("visualizations/model_accuracy.png")
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x=sorted_names, y=sorted_f1, palette="viridis")
plt.xticks(rotation=30)
plt.title("Model F1-Score Comparison (Sorted)")
plt.savefig("visualizations/model_f1_score.png")
plt.show()

# -----------------------------
# Final Print
# -----------------------------
print("\n=== Final Model Accuracies & F1 Scores ===")
for model in sorted_names:
    print(f"{model}: Accuracy={accuracies[model]:.4f}, F1={f1_scores[model]:.4f}")
    

# -----------------------------
# Generate PDF Report (Fixed)
# -----------------------------
from fpdf import FPDF

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Title
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "Customer Churn Prediction Report", ln=True)

# Selected Features
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 8, f"Selected Features ({len(selected_features)}): {', '.join(selected_features)}\n")

# Feature Importance Plot
pdf.image("visualizations/top_features.png", w=180)

# Add each model's results
for model in sorted_names:
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"{model} - Accuracy: {accuracies[model]:.4f}, F1 Score: {f1_scores[model]:.4f}", ln=True)
    
    cm_path = f"visualizations/confusion_matrix_{model.replace(' ', '_')}.png"
    roc_path = f"visualizations/roc_curve_{model.replace(' ', '_')}.png"
    
    if os.path.exists(cm_path):
        pdf.image(cm_path, w=120)
    if os.path.exists(roc_path):
        pdf.image(roc_path, w=120)

# Add Accuracy Comparison
pdf.add_page()
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "Accuracy Comparison", ln=True)
pdf.image("visualizations/model_accuracy.png", w=150)

# Add F1 Score Comparison
pdf.add_page()
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, "F1 Score Comparison", ln=True)
pdf.image("visualizations/model_f1_score.png", w=150)

# Save PDF
pdf.output("Churn_Model_Report.pdf")
print("✅ PDF Report saved as 'Churn_Model_Report.pdf'")


