import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

os.makedirs("outputs", exist_ok=True)

obj = joblib.load(os.path.join("models", "rf_iris.pkl"))
model = obj["model"]
feature_names = list(obj["feature_names"])
target_names = list(obj["target_names"])

data = load_iris()
X = data.data
y = data.target

df = pd.DataFrame(X, columns=feature_names)
df["species"] = [target_names[i] for i in y]

sns.pairplot(df, hue="species")
plt.savefig(os.path.join("outputs", "pairplot.png"))
plt.close()

clf = None
if hasattr(model, "named_steps") and "clf" in model.named_steps:
    clf = model.named_steps["clf"]
elif hasattr(model, "feature_importances_"):
    clf = model

if clf is not None and hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(8,4))
    plt.barh(range(len(importances)), importances[indices])
    plt.yticks(range(len(importances)), np.array(feature_names)[indices])
    plt.xlabel("importance")
    plt.title("feature importance")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "feature_importance.png"))
    plt.close()

desc = df.describe().reset_index().rename(columns={"index":"stat"})
desc = desc.rename(columns={c:c for c in desc.columns})
desc.to_csv(os.path.join("outputs", "feature_summary.csv"), index=False)
print("visualizations saved")
