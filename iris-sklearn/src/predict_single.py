import os
import joblib
import numpy as np

obj = joblib.load(os.path.join("models", "rf_iris.pkl"))
model = obj["model"]
target_names = list(obj["target_names"])

sample = np.array([[5.1,3.5,1.4,0.2]])
pred = model.predict(sample)[0]
probs = model.predict_proba(sample)[0]
print("predicted_index", int(pred))
print("predicted_species", str(target_names[int(pred)]))
print("probabilities", {target_names[i]: float(probs[i]) for i in range(len(probs))})
