import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_excel(r"C:\Users\WAJAHAT TRADERS\OneDrive - KSBL\Desktop\churn\RR-Train-forSPSS.xlsx", sheet_name = 'relay train')
test_df = pd.read_excel(r"C:\Users\WAJAHAT TRADERS\OneDrive - KSBL\Desktop\churn\RR-Test-forSPSS.xlsx", sheet_name = 'relay test.csv')

train_df.shape
test_df.shape
# Display basic info
train_df.info()
train_df.isna().sum()
train_df = train_df.dropna()



train_df['firstorder'] = pd.to_datetime(train_df['firstorder'],dayfirst=True, errors='coerce')
train_df['lastorder'] = pd.to_datetime(train_df['lastorder'],dayfirst=True, errors='coerce')

# Check for missing values after conversion
print("Missing Values':", train_df.isna().sum())
#%% train
import pandas as pd


# Preprocessing for train data
train_df = train_df.dropna()
train_df['firstorder'] = pd.to_datetime(train_df['firstorder'], dayfirst=True, errors='coerce')
train_df['lastorder'] = pd.to_datetime(train_df['lastorder'], dayfirst=True, errors='coerce')

train_df['firstorder'].fillna(train_df['firstorder'].median(), inplace=True)
train_df['lastorder'].fillna(train_df['lastorder'].median(), inplace=True)
train_df['created'].fillna(train_df['created'].median(), inplace=True)

train_df['days_since_firstorder'] = (pd.Timestamp.now() - pd.to_datetime(train_df['firstorder'])).dt.days
train_df['days_since_lastorder'] = (pd.Timestamp.now() - pd.to_datetime(train_df['lastorder'])).dt.days
train_df['days_since_created'] = (pd.Timestamp.now() - pd.to_datetime(train_df['created'])).dt.days
train_df['days_between_orders'] = (pd.to_datetime(train_df['lastorder']) - pd.to_datetime(train_df['firstorder'])).dt.days
train_df['avg_days_per_order'] = train_df['days_between_orders'] / train_df['ordfreq']
train_df['avg_days_per_order'].replace([float('inf'), -float('inf')], 0, inplace=True)
train_df['avg_days_per_order'].fillna(0, inplace=True)

train_df['tenure'] = train_df['days_since_firstorder']
train_df['recency_ratio'] = train_df['days_since_lastorder'] / train_df['tenure']
train_df['recency_ratio'].replace([float('inf'), -float('inf')], 0, inplace=True)
train_df['recency_ratio'].fillna(0, inplace=True)
train_df['days_since_firstorder_created'] = train_df['days_since_created'] - train_df['days_since_firstorder']


#%% test
# Now, preprocess the test data in the same way
test_df['firstorder'] = pd.to_datetime(test_df['firstorder'], dayfirst=True, errors='coerce')
test_df['lastorder'] = pd.to_datetime(test_df['lastorder'], dayfirst=True, errors='coerce')

test_df['firstorder'].fillna(test_df['firstorder'].median(), inplace=True)
test_df['lastorder'].fillna(test_df['lastorder'].median(), inplace=True)
test_df['created'].fillna(test_df['created'].median(), inplace=True)

test_df['days_since_firstorder'] = (pd.Timestamp.now() - pd.to_datetime(test_df['firstorder'])).dt.days
test_df['days_since_lastorder'] = (pd.Timestamp.now() - pd.to_datetime(test_df['lastorder'])).dt.days
test_df['days_since_created'] = (pd.Timestamp.now() - pd.to_datetime(test_df['created'])).dt.days
test_df['days_between_orders'] = (pd.to_datetime(test_df['lastorder']) - pd.to_datetime(test_df['firstorder'])).dt.days
test_df['avg_days_per_order'] = test_df['days_between_orders'] / test_df['ordfreq']
test_df['avg_days_per_order'].replace([float('inf'), -float('inf')], 0, inplace=True)
test_df['avg_days_per_order'].fillna(0, inplace=True)

test_df['tenure'] = test_df['days_since_firstorder']
test_df['recency_ratio'] = test_df['days_since_lastorder'] / test_df['tenure']
test_df['recency_ratio'].replace([float('inf'), -float('inf')], 0, inplace=True)
test_df['recency_ratio'].fillna(0, inplace=True)
test_df['days_since_firstorder_created'] = test_df['days_since_created'] - test_df['days_since_firstorder']

# Days since first order relative to days created
"""
# Frequency feature engineering
train_df['order_frequency_ratio'] = train_df['ordfreq'] / train_df['tenure']
test_df['order_frequency_ratio'] = test_df['ordfreq'] / test_df['tenure']

# Replace infinite or NaN values in order_frequency_ratio
train_df['order_frequency_ratio'].replace([float('inf'), -float('inf')], 0, inplace=True)
train_df['order_frequency_ratio'].fillna(0, inplace=True)
"""
# Verify the new features
print(train_df[['days_since_created', 'days_since_firstorder', 'days_since_lastorder', 'days_between_orders', 
                'avg_days_per_order', 'tenure', 'recency_ratio']].head())
##
# %%# Declare feature vector and target variable
X_train = train_df.drop(['custid','retained','created','firstorder','lastorder','train'], axis=1)
X_test = test_df.drop(['custid', 'created', 'firstorder', 'lastorder', 'retained','train'], axis=1)
y_train = train_df['retained']
y_test = test_df['retained']

# OR - Split data into separate training and test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

#%%
# ENCODE
import category_encoders as ce
encoder = ce.OneHotEncoder(cols=['favday','city'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
X_train.columns
#
from sklearn.preprocessing import StandardScaler
# Initialize StandardScaler
scaler = StandardScaler()
# List of continuous columns
continuous_columns = [
    "esent", "eopenrate", "eclickrate", "avgorder", "ordfreq", 
    "days_since_firstorder", "days_since_lastorder", "days_since_created",
    "days_between_orders", "avg_days_per_order", "tenure", "recency_ratio",
    "days_since_firstorder_created"
]

# Scale only continuous features
X_train[continuous_columns] = scaler.fit_transform(X_train[continuous_columns])
X_test[continuous_columns] = scaler.transform(X_test[continuous_columns])
#%% LOGESTIC REGRESSION
# Initialize Logistic Regression model
from sklearn.model_selection import GridSearchCV # train,tst,sp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

logistic_model = LogisticRegression(max_iter=1000)

# Hyperparameter tuning using GridSearchCV
param_grid_lr = {
    'C': [0.1, 1, 10],  # Regularization strength
    'penalty': ['l1', 'l2'],  # L1 or L2 regularization
    'solver': ['liblinear', 'saga'],  # Solvers that support L1 regularization
    'max_iter': [1000]  # Number of iterations
}

# GridSearchCV to find the best hyperparameters
grid_search_lr = GridSearchCV(logistic_model, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_logistic_model = grid_search_lr.best_estimator_

# Make predictions with the best model
y_pred_lr = best_logistic_model.predict(X_test)

# Evaluate the best model
print("Best Hyperparameters:", grid_search_lr.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

"""
logistic_model = LogisticRegression()

# Train the model
logistic_model.fit(X_train, y_train)

# Make predictions
y_pred = logistic_model.predict(X_test)
"""
#%% Random Forest
"""rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Predictions for test data:", y_pred_rf)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))"""
# Initialize Random Forest model
rf_model = RandomForestClassifier(random_state=42)
# Ensure you have enough trees (n_estimators) for stability and avoid overfitting with appropriate max_depth and min_samples_split.
# Hyperparameter tuning using GridSearchCV
param_grid_rf = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest # for Stability
    'max_depth': [10, 20, 30, None],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider for the best split
    'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
}

# GridSearchCV to find the best hyperparameters
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5 , scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_rf_model = grid_search_rf.best_estimator_

# Make predictions with the best model
y_pred_rf = best_rf_model.predict(X_test)

# Evaluate the best model
print("Best Hyperparameters:", grid_search_rf.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
#%%
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize XGBClassifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Define hyperparameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [3, 5, 7],          # Maximum depth of a tree
    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
    'subsample': [0.8, 1.0],          # Subsample ratio of the training data
    'colsample_bytree': [0.8, 1.0],   # Subsample ratio of columns when constructing each tree
}

# Perform GridSearchCV for XGBoost
grid_search_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid_xgb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid_search_xgb.fit(X_train, y_train)

# Get the best XGBoost model
best_xgb_model = grid_search_xgb.best_estimator_

# Make predictions with the best XGBoost model
y_pred_xgb = best_xgb_model.predict(X_test)

# Evaluate the XGBoost model
print("Best Hyperparameters for XGBoost:", grid_search_xgb.best_params_)
print("Accuracy for XGBoost:", accuracy_score(y_test, y_pred_xgb))
print("\nClassification Report for XGBoost:\n", classification_report(y_test, y_pred_xgb))
print("\nConfusion Matrix for XGBoost:\n", confusion_matrix(y_test, y_pred_xgb))



#%%
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize MLPClassifier
nn_model = MLPClassifier(max_iter=1000, random_state=42)

# Hyperparameter tuning using GridSearchCV for Neural Network
param_grid_nn = {
    'hidden_layer_sizes': [(50,), (100,), (150,)],  # Number of neurons in each hidden layer
    'activation': ['relu', 'tanh'],  # Activation functions for the hidden layers
    'solver': ['adam', 'lbfgs'],  # Optimizer
    'alpha': [0.0001, 0.001, 0.01],  # Regularization strength
}

# GridSearchCV for Neural Network
grid_search_nn = GridSearchCV(nn_model, param_grid_nn, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_nn.fit(X_train, y_train) 

# Get the best Neural Network model
best_nn_model = grid_search_nn.best_estimator_

# Make predictions with the best Neural Network model
y_pred_nn = best_nn_model.predict(X_test)

# Evaluate the Neural Network model
print("Best Hyperparameters for Neural Network:", grid_search_nn.best_params_)
print("Accuracy for Neural Network:", accuracy_score(y_test, y_pred_nn))
print("\nClassification Report for Neural Network:\n", classification_report(y_test, y_pred_nn))
print("\nConfusion Matrix for Neural Network:\n", confusion_matrix(y_test, y_pred_nn))
#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Function to plot and save confusion matrix
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Churn", "Churn"], yticklabels=["Not Churn", "Churn"])
    plt.title(f"Confusion Matrix: {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{filename}.png")
    plt.show()

# Plot and save confusion matrices for all models
plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression", "confusion_matrix_lr")
plot_confusion_matrix(y_test, y_pred_rf, "Random Forest", "confusion_matrix_rf")
plot_confusion_matrix(y_test, y_pred_nn, "Neural Network", "confusion_matrix_nn")

# Create a comparison of accuracies
model_names = ["Logistic Regression", "Random Forest", "Neural Network"]
accuracies = [
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_rf),
    accuracy_score(y_test, y_pred_nn)
]

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=accuracies, palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.show()

# Save classification reports to a file
with open("classification_reports.txt", "w") as f:
    f.write("Logistic Regression Report\n")
    f.write(classification_report(y_test, y_pred_lr))
    f.write("\nRandom Forest Report\n")
    f.write(classification_report(y_test, y_pred_rf))
    f.write("\nNeural Network Report\n")
    f.write(classification_report(y_test, y_pred_nn))

# Save best hyperparameters for documentation
best_hyperparameters = {
    "Logistic Regression": grid_search_lr.best_params_,
    "Random Forest": grid_search_rf.best_params_,
    "Neural Network": grid_search_nn.best_params_
}
pd.DataFrame(best_hyperparameters).to_csv(r"C:\Users\WAJAHAT TRADERS\OneDrive - KSBL\Desktop\churn\best_hyperparameters.csv")
#%%
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", size=12)
        self.cell(0, 10, "Churn Prediction Report", ln=True, align="C")

    def add_section(self, title, text):
        self.set_font("Arial", size=10)
        self.cell(0, 10, title, ln=True, align="L")
        self.multi_cell(0, 10, text)
        self.ln()

    def add_image(self, image_path, w=150):
        self.image(image_path, x=(210 - w) / 2, w=w)
        self.ln()

# Create a PDF report
pdf = PDFReport()
pdf.add_page()

# Add text sections
pdf.add_section("Introduction", "This report summarizes model performance for churn prediction.")
pdf.add_section("Logistic Regression Report", classification_report(y_test, y_pred_lr))
pdf.add_section("Random Forest Report", classification_report(y_test, y_pred_rf))
pdf.add_section("Neural Network Report", classification_report(y_test, y_pred_nn))

# Add images
pdf.add_image("confusion_matrix_lr.png")
pdf.add_image("confusion_matrix_rf.png")
pdf.add_image("confusion_matrix_nn.png")
pdf.add_image("model_accuracy_comparison.png")

# Save the report
pdf.output(r"C:\Users\WAJAHAT TRADERS\OneDrive - KSBL\Desktop\churn\churn_prediction_report.pdf")

#%%
# Check correlations
#correlations = train_df.corr()
#sns.heatmap(correlations, cmap='coolwarm', annot=False)
#plt.show()#
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation matrix for numerical columns
#num_df = ed_train.select_dtypes(include=np.number)
#correlations = num_df.corr()
correlations = X_train.corr()
# Mask values that are < 0.10 (set them to NaN to exclude them from the plot)
correlations_masked = correlations.where(correlations >= 0.10, np.nan)

# Set up the figure size to ensure the heatmap fits well
plt.figure(figsize=(12, 8))

# Plot the heatmap with the optimal options
sns.heatmap(correlations_masked, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5, cbar=True, square=True)

# Rotate the labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45, va='top')

# Add title
plt.title('Correlation Matrix', fontsize=16)

# Show the plot
plt.tight_layout()
plt.show()

#%%

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Feature Importance
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
importances.sort_values().plot(kind='barh', figsize=(10, 6))
plt.title("Feature Importance")
plt.show()


import os
print(os.getcwd())

#%%
# f_df 
train = ed_train.copy()
test = ed_test.copy()
# Convert datetime to numeric (days since some reference date)


# Extracting only the relevant features for churn prediction
selected_columns = [
   ['retained', 'esent', 'eopenrate',
          'eclickrate', 'avgorder', 'ordfreq', 'paperless', 'refill', 'doorstep',
          , 'favday_1', 'favday_2', 'favday_3', 'favday_4', 'favday_5',
          'favday_6', 'favday_7', 'city_1', 'city_2', 'city_3', 'city_4',
          'days_since_created', 'days_between_orders', 'days_since_firstorder',
          'days_since_lastorder', 'days_diff']

train_selected = train[selected_columns]
test_selected = test[selected_columns]

# Check the shape of the selected features
print(train_selected.shape)
print(test_selected.shape)






















def vab(correlation_matrix, target_column, threshold=0.10):
    selected_columns = []  # To store the column names with correlation >= threshold
    for column in correlation_matrix.columns:
        # If the absolute correlation with the target column is greater than or equal to the threshold
        if abs(correlation_matrix.loc[target_column, column]) >= threshold:
            selected_columns.append(column)  # Add the column name to the list
    return selected_columns

# Example usage
correlations = train_df.select_dtypes(include=np.number).corr()  # Calculate correlation matrix
target_column = 'retained'  # Assuming 'retained' is the target column for churn prediction

selected_columns = vab(correlations, target_column, threshold=0.10)
print("Selected columns based on correlation threshold:", selected_columns)













# Check missing values
print("Missing Values:\n", train_df.isnull().sum())

"""import re
# Function to standardize date format
def standardize_date(date_str):
    try:
        # Match and rearrange as YYYY-MM-DD
        standardized_date = re.sub(r'(\d{2})/(\d{2})/(\d{4})', r'\3-\2-\1', date_str)
        return standardized_date
    except:
        return None
"""
# Apply the function to the column
#train_df['firstorder'] = train_df['firstorder'].apply(standardize_date)
# Impute with median date
# Convert 'firstorder' to datetime with dayfirst=True
train_df['firstorder'] = pd.to_datetime(train_df['firstorder'],dayfirst=True, errors='coerce')
train_df['lastorder'] = pd.to_datetime(train_df['lastorder'],dayfirst=True, errors='coerce')

# Check for missing values after conversion
print("Missing Values':", train_df.isna().sum())
train_df['firstorder'].fillna(train_df['firstorder'].median(),inplace = True)
train_df['lastorder'].fillna(train_df['lastorder'].median(),inplace = True)
train_df['created'].fillna(train_df['created'].median(),inplace = True)

