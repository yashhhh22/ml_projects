import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

model_path = os.path.join("models", "mnist_cnn.pt")
if os.path.exists(model_path) and ("--force" not in sys.argv):
    print(f"model found at {model_path}, skipping training (run: python src/train.py --force to retrain)")
    sys.exit(0)
    
def build_model(device):
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = torch.nn.MaxPool2d(2,2)
            self.dropout1 = torch.nn.Dropout(0.25)
            self.fc1 = torch.nn.Linear(64*14*14, 128)
            self.dropout2 = torch.nn.Dropout(0.5)
            self.fc2 = torch.nn.Linear(128, 10)
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.dropout1(x)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            return x
    return Net().to(device)

def evaluate_loader(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item() * data.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += data.size(0)
    return running_loss/total, correct/total

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    train_idx, val_idx = train_test_split(np.arange(len(train_dataset)), test_size=0.1, random_state=seed, stratify=train_dataset.targets.numpy())
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset, val_idx)
    batch_size = 128
    workers = 0 if os.name == "nt" else 2
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 5
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += data.size(0)
        train_loss = running_loss/total
        train_acc = correct/total
        val_loss, val_acc = evaluate_loader(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join("models", "mnist_cnn.pt"))
        print(f"epoch {epoch} train_loss {train_loss:.4f} train_acc {train_acc:.4f} val_loss {val_loss:.4f} val_acc {val_acc:.4f}")
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(os.path.join("outputs", "training_history.csv"), index=False)
    artifact = {"state_dict_path": os.path.join("models", "mnist_cnn.pt"), "history_csv": os.path.join("outputs", "training_history.csv")}
    joblib.dump(artifact, os.path.join("models", "mnist_artifact.pkl"))
    print("training complete")

if __name__ == "__main__":
    main()
