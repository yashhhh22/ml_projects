Iris Flower Classification – Scikit-Learn (Full ML Pipeline)

A complete classical machine learning project built using scikit-learn, featuring dataset preprocessing, training, model selection using GridSearchCV, evaluation, visualization, and interactive prediction.
This project follows a professional, modular ML pipeline structure.

📁 Project Structure
iris-sklearn/
│
├── src/
│   ├── train.py              # trains model with pipeline + grid search
│   ├── evaluate.py           # generates metrics, predictions CSV, confusion matrix
│   ├── visualize.py          # creates feature importance, pairplot, feature summary
│   ├── predict_single.py     # loads model & predicts a predefined sample
│   ├── predict_cli.py        # CLI allowing user-input numeric features
│   └── run_all.py            # master script: train → evaluate → visualize
│
├── models/
│   └── rf_iris.pkl           # saved RandomForest model
│
├── outputs/
│   ├── predictions.csv
│   ├── metrics_summary.csv
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── pairplot.png
│   ├── learning_curve.png
│   └── feature_summary.csv
│
├── requirements.txt
├── README.md
└── .gitignore

🛠 Technologies Used
Python 3
Scikit-Learn – model training & evaluation
Pandas / NumPy – dataset handling
Matplotlib / Seaborn – visualization
Joblib – model serialization

🚀 Key Features

✔ End-to-End ML Pipeline
Preprocessing with StandardScaler
Model building with RandomForestClassifier
Hyperparameter tuning with GridSearchCV
Saving the trained model in models/
Generating metrics, CSVs, and plots

✔ Interpretability
Feature importance
Pairplot visualizations
Summary statistics CSV

✔ Interactive Predictions
CLI prediction from user input
Single-sample prediction script

🧑‍💻 How to Run (VS Code Integrated Terminal)

1. Open project folder
iris-sklearn/

2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Run complete ML pipeline
python src/run_all.py

This performs:
Training (pipeline + grid search)
Evaluation (predictions, confusion matrix)
Visualizations

5. Run individual scripts
python src/train.py
python src/evaluate.py
python src/visualize.py

6. CLI prediction
python src/predict_cli.py

7. Single predefined sample prediction
python src/predict_single.py

📂 What to Include When Submitting

Upload these items:
✔ Model:
models/rf_iris.pkl

✔ Output artifacts:
outputs/predictions.csv
outputs/confusion_matrix.png
outputs/feature_importance.png
outputs/learning_curve.png
outputs/pairplot.png

✔ Source code:
Entire src/ folder

✔ Docs:
README.md
requirements.txt

This ensures your submission is complete, reproducible, and professional.

📘 Technical Summary

train.py builds a scikit-learn Pipeline with:
StandardScaler
RandomForestClassifier
Hyperparameter tuning via GridSearchCV (cross-validation).

evaluate.py computes:
Classification report
Test accuracy
Learning curves
Confusion matrix
Predictions CSV

visualize.py outputs:
Seaborn pairplot
Feature importance chart
Feature distribution statistics

The entire workflow follows real ML engineering standards.