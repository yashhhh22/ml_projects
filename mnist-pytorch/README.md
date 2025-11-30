MNIST CNN Classification – PyTorch (Full ML Pipeline)

A complete, production-style deep learning project built using PyTorch, featuring dataset loading, model training, evaluation, visualization, and interactive prediction.
This repository demonstrates real ML engineering practices with clear modular structure.

🔧 Project Structure
mnist-pytorch/
│
├── src/
│   ├── train.py              # trains CNN, saves best model
│   ├── evaluate.py           # evaluates saved model, outputs classification report, confusion matrix, predictions CSV
│   ├── visualize.py          # generates loss/accuracy curves and sample predictions
│   ├── predict_cli.py        # interactive CLI prediction on test samples
│   ├── predict_image.py      # predicts digit from a custom image
│   └── run_all.py            # runs full pipeline: train → evaluate → visualize
│
├── models/                   # stores trained model artifacts
│   ├── mnist_cnn.pt
│   └── mnist_artifact.pkl
│
├── outputs/                  # saved metrics, CSVs, prediction images, graphs
│   ├── training_history.csv
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   ├── classification_report.csv
│   └── predictions.csv
│
├── sample_3.png
├── sample_8.png
│
├── requirements.txt
├── README.md
└── .gitignore

🚀 Technologies Used
1. Python 3
2. PyTorch – deep learning framework
3. Torchvision – dataset & transforms
4. NumPy / Pandas – data handling
5. Matplotlib – visualizations
6. tqdm – training progress bars
7. joblib – artifact saving

📌 Key Features
✔ Complete ML Pipeline
Automated training
Validation monitoring
Best-model checkpointing
Evaluation using metrics + confusion matrix
Visualization of learning curves

✔ Interactive Prediction
Predict digits using test indexes (predict_cli.py)
Predict handwritten digits from custom images (predict_image.py)

✔ Modular Engineering
Each component (train, evaluate, visualize, inference) is separated for clarity and reusability.

🧑‍💻 How to Run (VS Code Integrated Terminal)
1. Open folder
mnist-pytorch/

2. (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Run full pipeline
python src/run_all.py

This will:
Train the CNN
Save best model to models/mnist_cnn.pt
Evaluate the model
Generate all visualizations

5. Run individual scripts
python src/train.py
python src/evaluate.py
python src/visualize.py

6. Predict using CLI
python src/predict_cli.py

7. Predict from custom image
python src/predict_image.py path/to/image.png

📂 Outputs Generated in outputs/
training_history.csv
accuracy_curve.png
loss_curve.png
confusion_matrix.png
classification_report.csv
predictions.csv
sample_predictions.png

All visualizations and CSV logs are auto-generated on evaluation.

📝 Model Artifacts

Saved inside models/:
mnist_cnn.pt – best checkpoint
mnist_artifact.pkl – metadata + history path

🏁 Project Summary

This project contains:
A professional ML pipeline
A CNN model for digit classification
Fully reproducible training
Evaluation metrics and graphs
CLI + image prediction systems
Proper folder structure and modular code

Ideal for ML assignments, academic submissions, interviews, and GitHub portfolios