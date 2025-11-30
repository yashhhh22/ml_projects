import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.datasets import load_iris

os.makedirs("outputs", exist_ok=True)

obj = joblib.load(os.path.join("models", "rf_iris.pkl"))
model = obj["model"]
feature_names = list(obj["feature_names"])
target_names = list(obj["target_names"])

data = load_iris()
X = data.data
y = data.target

train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring="accuracy", n_jobs=-1, train_sizes=np.linspace(0.1,1.0,5))
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, marker="o", label="train")
plt.plot(train_sizes, test_mean, marker="o", label="validation")
plt.xlabel("training examples")
plt.ylabel("accuracy")
plt.legend()
plt.title("Learning Curve")
plt.savefig(os.path.join("outputs", "learning_curve.png"))
plt.close()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
df_report = pd.DataFrame(report).transpose().reset_index().rename(columns={"index":"class"})
cols = ["class", "precision", "recall", "f1-score", "support"]
for c in cols:
    if c not in df_report.columns:
        df_report[c] = ""
df_report = df_report[cols]
df_report.to_csv(os.path.join("outputs", "classification_report.csv"), index=False)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=target_names, yticklabels=target_names, cmap="Blues")
plt.xlabel("predicted")
plt.ylabel("true")
plt.title("Confusion Matrix")
plt.savefig(os.path.join("outputs", "confusion_matrix.png"))
plt.close()

pred_df = pd.DataFrame(X_test, columns=feature_names)
pred_df["true_label"] = [target_names[i] for i in y_test]
pred_df["pred_label"] = [target_names[i] for i in y_pred]
pred_df.to_csv(os.path.join("outputs", "predictions.csv"), index=False)
print("evaluation complete")
