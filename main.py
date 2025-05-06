# Heart Disease Prediction using Random Forest
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, RocCurveDisplay, r2_score, mean_squared_error
)

# Load dataset
df = pd.read_csv("heart_disease_uci.csv")

# Drop unused columns
df_clean = df.drop(columns=["id", "dataset"]).copy()

# Handle missing numeric data
num_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns
num_imputer = SimpleImputer(strategy="median")
df_clean[num_cols] = num_imputer.fit_transform(df_clean[num_cols])

# Encode categorical columns
cat_cols = df_clean.select_dtypes(include="object").columns
for col in cat_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

# Split features and targets
X = df_clean.drop(columns="num")
y = df_clean["num"]
y_binary = (y > 0).astype(int)  # Binary: 0 (no disease), 1 (disease)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Evaluation - Classification
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Plot ROC curve
RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.title("ROC Curve - Random Forest Classifier")
plt.show()

# Feature Importances
feat_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": clf.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_importances, x="Importance", y="Feature", palette="viridis")
plt.title("Feature Importances - Random Forest")
plt.tight_layout()
plt.show()

# Regression - Predict severity level (0 to 4)
y_severity = df_clean["num"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_severity, test_size=0.2, random_state=42)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)

print("R² Score (Regression):", r2_score(y_test_r, y_pred_r))
print("RMSE (Regression):", np.sqrt(mean_squared_error(y_test_r, y_pred_r)))
