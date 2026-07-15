"""
==========================================================
Using XGBoost to classify between malignant and benign
==========================================================

+ The parameters we got from hyperparameter_tuning.py: 
- n_estimator=200, 
- max_depth=3, 
- learning_rate=0.2
 we will use them in our model.

**********************************************************
Results : 
**********************************************************
Data shape: (569, 30)

Accuracy: 0.956140350877193

AUC: 0.9897486772486772

Classification report:
              precision    recall  f1-score   support

   malignant       0.97      0.90      0.94        42
      benign       0.95      0.99      0.97        72

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114


Confusion matrix:
[[38  4]
 [ 1 71]]

Top 10 important features:
  worst radius                   0.2189
  mean concave points            0.1888
  worst perimeter                0.1860
  worst concave points           0.1032
  worst concavity                0.0494
  worst area                     0.0394
  mean area                      0.0244
  mean texture                   0.0204
  worst smoothness               0.0197
  concavity error                0.0180
**********************************************************
"""
# Imports
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import xgboost as xgb
import numpy as np

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.target_names

print(f"Data shape: {X.shape}")

# train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build model
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)

# Train model
model.fit(X_train, y_train, eval_set=[ (X_test, y_test)], verbose=False)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\nAccuracy: {acc}")
print(f"\nAUC: {auc}")
print(f"\nClassification report:")
print(classification_report(y_test, y_pred, target_names=feature_names))
print(f"\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))


# Feature importance (top 10)
importances = model.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]
print("\nTop 10 important features:")
for i in top_idx:
    print(f"  {data.feature_names[i]:30s} {importances[i]:.4f}")

# Save the model
model.save_model("xgb_breast_cancer.json")
