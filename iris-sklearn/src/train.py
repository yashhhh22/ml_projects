import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42))
])

param_grid = {
    "clf__n_estimators": [50, 100, 200],
    "clf__max_depth": [None, 3, 5],
    "clf__min_samples_split": [2, 4]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)

gs.fit(X_train, y_train)

best_model = gs.best_estimator_

artifact = {
    "model": best_model,
    "feature_names": feature_names,
    "target_names": target_names,
    "best_params": gs.best_params_
}

joblib.dump(artifact, os.path.join("models", "rf_iris.pkl"))

train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)

metrics_df = pd.DataFrame([{
    "train_accuracy": float(train_score),
    "test_accuracy": float(test_score)
}])
metrics_df.to_csv(os.path.join("outputs", "metrics_summary.csv"), index=False)
print("best_params", gs.best_params_)
print("train_accuracy", train_score, "test_accuracy", test_score)
