import os
import sys
import joblib
import numpy as np

MODEL_PATH = os.path.join("models", "rf_iris.pkl")
if not os.path.exists(MODEL_PATH):
    print("model_missing")
    sys.exit(1)

obj = joblib.load(MODEL_PATH)
model = obj["model"]
target_names = list(obj["target_names"])

def safe_float(x):
    try:
        return float(x)
    except:
        return None

def input_sample():
    raw = input("enter four values sepal_length sepal_width petal_length petal_width or type exit: ").strip()
    if raw.lower() == "exit":
        return "exit"
    parts = raw.split()
    if len(parts) != 4:
        print("invalid_count")
        return None
    vals = [safe_float(p) for p in parts]
    if any(v is None for v in vals):
        print("invalid_number")
        return None
    return np.array(vals, dtype=float).reshape(1, -1)

print("iris_cli_ready")

while True:
    s = input_sample()
    if isinstance(s, str) and s == "exit":
        break
    if s is None:
        continue
    pred = model.predict(s)[0]
    probs = model.predict_proba(s)[0]
    print("predicted_index", int(pred))
    print("predicted_species", target_names[int(pred)])
    print("probabilities", {target_names[i]: float(probs[i]) for i in range(len(probs))})

print("bye")
