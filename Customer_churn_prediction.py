import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
# Random Forest Feature Selection
# -----------------------------
rf_selector = RandomForestClassifier(n_estimators=200, random_state=42)
rf_selector.fit(X_train, y_train)

# Get feature importance
importances = pd.Series(rf_selector.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# ✅ Fixed Seaborn palette warning
plt.figure(figsize=(10, 6))
sns.barplot(
    x=importances.head(15),
    y=importances.head(15).index,
    hue=importances.head(15).index,  # set hue to y
    palette="viridis",
    dodge=False,
    legend=False
)
plt.title("Top 15 Features by Random Forest Importance")
plt.show()

# -----------------------------
# Select Features Above Threshold
# -----------------------------
threshold = 0.005
selected_features = importances[importances >= threshold].index.tolist()
print(f"Selected Features ({len(selected_features)}):", selected_features)

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
# Model Training Function
# -----------------------------
def train_and_evaluate(name, model, params):
    grid = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{name} Best Params:", grid.best_params_)
    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    return acc

# -----------------------------
# Logistic Regression
# -----------------------------
acc_lr = train_and_evaluate(
    "Logistic Regression",
    LogisticRegression(max_iter=1000),
    {'C':[0.1, 1, 10], 'penalty':['l1','l2'], 'solver':['liblinear','saga']}
)

# -----------------------------
# Random Forest
# -----------------------------
acc_rf = train_and_evaluate(
    "Random Forest",
    RandomForestClassifier(random_state=42),
    {'n_estimators':[100,200], 'max_depth':[10,20,None], 'min_samples_split':[2,5]}
)

# -----------------------------
# Neural Network
# -----------------------------
acc_nn = train_and_evaluate(
    "Neural Network",
    MLPClassifier(max_iter=500, random_state=42),
    {'hidden_layer_sizes':[(50,),(100,)], 'activation':['relu','tanh'], 'alpha':[0.0001,0.001]}
)

# -----------------------------
# Accuracy Comparison
# -----------------------------
model_names = ["Logistic Regression", "Random Forest", "Neural Network"]
accuracies = [acc_lr, acc_rf, acc_nn]
sns.barplot(x=model_names, y=accuracies, palette="mako")
plt.title("Model Accuracy Comparison")
plt.show()
