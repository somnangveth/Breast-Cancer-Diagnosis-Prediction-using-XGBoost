"""
======================================================================
After Training our model we are going to test them:
======================================================================

These are the values we are going to enter:

Feature     Value
mean radius               =       13.54
mean texture              =       14.36
mean perimeter            =       87.46
mean area                 =       566.3
mean smoothness           =       0.09779
mean compactness          =       0.08129
mean concavity            =       0.06664
mean concave points       =       0.04781
mean symmetry             =       0.1885
mean fractal dimension    =       0.05766
radius error              =       0.2699
texture error             =       0.7886
perimeter error           =       2.058
area error                =       23.56
smoothness error          =       0.008462
compactness error         =       0.0146
concavity error           =       0.02387
concave points error      =       0.01315
symmetry error            =       0.0198
fractal dimension error   =       0.0023
worst radius              =       15.11
worst texture             =       19.26
worst perimeter           =       99.7
worst area                =       711.2
worst smoothness          =       0.144
worst compactness         =       0.1773
worst concavity           =       0.239
worst concave points      =       0.1288
worst symmetry            =       0.2977
worst fractal dimension   =       0.07259


======================================================================
Result:
======================================================================
Enter the following 30 values one by one.


--- Prediction ---
Predicted class : benign
P(malignant)    : 0.0039
P(benign)       : 0.9961

"""

# Testing our model
import xgboost as xgb
from sklearn.datasets import load_breast_cancer

# Feature names, in the exact order the model expects
feature_names = load_breast_cancer().feature_names

# Load trained model
model = xgb.XGBClassifier()
model.load_model("xgb_breast_cancer.json")

print("Enter the following 30 values one by one.\n")

values = []
for name in feature_names:
    while True:
        raw = input(f"{name}: ")
        try:
            values.append(float(raw))
            break
        except ValueError:
            print("  Please enter a valid number.")

# Reshape into a single-row 2D array (model expects shape [n_samples, n_features])
X_new = [values]

pred = model.predict(X_new)[0]
proba = model.predict_proba(X_new)[0]

label = "benign" if pred == 1 else "malignant"

print("\n--- Prediction ---")
print(f"Predicted class : {label}")
print(f"P(malignant)    : {proba[0]:.4f}")
print(f"P(benign)       : {proba[1]:.4f}")
