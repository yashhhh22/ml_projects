import os
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def build_model(device):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2,2)
            self.dropout1 = nn.Dropout(0.25)
            self.fc1 = nn.Linear(64*14*14, 128)
            self.dropout2 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, 10)
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.dropout1(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            return x
    return Net().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def evaluate_loop(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Evaluating"):
            data = data.to(device)
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_probs.extend(probs.cpu().numpy().tolist())
            y_true.extend(target.numpy().tolist())
    return y_true, y_pred, y_probs

def main():
    os.makedirs("outputs", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    workers = 0 if os.name == "nt" else 2
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=workers, pin_memory=False)
    artifact_path = os.path.join("models", "mnist_artifact.pkl")
    if not os.path.exists(artifact_path):
        print("model_missing")
        return
    artifact = joblib.load(artifact_path)
    state_dict_path = artifact["state_dict_path"]
    model = build_model(device)
    state = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state)
    y_true, y_pred, y_probs = evaluate_loop(model, test_loader, device)
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose().reset_index().rename(columns={"index":"class"})
    cols = ["class", "precision", "recall", "f1-score", "support"]
    for c in cols:
        if c not in df_report.columns:
            df_report[c] = ""
    df_report = df_report[cols]
    df_report.to_csv(os.path.join("outputs", "classification_report.csv"), index=False)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title("confusion_matrix")
    plt.savefig(os.path.join("outputs", "confusion_matrix.png"))
    plt.close()
    predictions_df = pd.DataFrame(y_probs, columns=[str(i) for i in range(10)])
    predictions_df["true_label"] = y_true
    predictions_df["pred_label"] = y_pred
    predictions_df.to_csv(os.path.join("outputs", "predictions.csv"), index=False)
    print("evaluation complete")

if __name__ == "__main__":
    main()
