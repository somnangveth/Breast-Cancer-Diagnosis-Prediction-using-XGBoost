"""
========================================================================
Tune XGBoost Hyperparameter using GridSearchCV
- By using gridsearchcv we can determine the optimal paramters to use
  for our model.
========================================================================
Result we got: 

Fitting 5 folds for each of 80 candidates, totalling 400 fits

Best parameters found: 
{'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 200}

Best cross-validation ROC AUC: 0.9911
Test set accuracy with best params:  0.9649
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb

# 1. Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# 2. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define the grid of values to try
param_grid = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
}

base_model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)

# cv=5 means 5-fold cross-validation on the training data
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)

print("\nBest parameters found: ")
print(grid_search.best_params_)
print(f"\nBest cross-validation ROC AUC: {grid_search.best_score_:.4f}")

#Evaluate the best model on the held-out test set
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test set accuracy with best params: {test_score: .4f}")
