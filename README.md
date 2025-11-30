Machine Learning Projects:

This repository contains a collection of fully engineered Machine Learning and Deep Learning projects, each built with a professional modular pipeline, clean directory structure, 
reproducible training workflow, and interactive prediction interfaces.

Each project demonstrates an end-to-end ML lifecycle including:
Data loading & preprocessing
Training
Evaluation
Visualization
Saving model artifacts
CLI-based prediction systems

Perfect for academics, viva exams, portfolios, and ML engineering practice.

📂 Projects Included
1️⃣ Iris Classification — Scikit-Learn

A complete classical ML pipeline built on the Iris dataset, implemented using Scikit-Learn Pipelines and GridSearchCV.
Features:
StandardScaler + RandomForest inside a unified Pipeline
Hyperparameter tuning using GridSearchCV
Evaluation with classification report & confusion matrix
Learning curve analysis
Feature importance visalization
Pairplot-based EDA
CLI prediction interface
Single-sample prediction script
Model and metric artifacts saved in organized folders

📁 Folder
iris-sklearn/

2️⃣ MNIST Digit Classification — PyTorch

A deep learning project that trains a Convolutional Neural Network (CNN) on the MNIST handwritten digits dataset using PyTorch.
Features:
Custom CNN architecture
Training loop with accuracy & loss tracking
Validation split with best-model checkpointing
Evaluation on test set with metrics and confusion matrix
Loss/accuracy curve visualization
Sample prediction grid
CLI predictor (sample index or image path)
Script to classify custom images
Saved model + training artifact for reproducibility

📁 Folder
mnist-pytorch/

🛠️ Technologies Used
Python 3
Scikit-Learn
PyTorch + Torchvision
NumPy
Pandas
Matplotlib, Seaborn
tqdm
joblib
