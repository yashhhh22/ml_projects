import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

os.makedirs("outputs", exist_ok=True)

artifact = joblib.load(os.path.join("models", "mnist_artifact.pkl"))
hist_path = artifact["history_csv"]
df_hist = pd.read_csv(hist_path)

plt.figure()
plt.plot(df_hist["train_loss"], label="train_loss")
plt.plot(df_hist["val_loss"], label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title("loss_curve")
plt.savefig(os.path.join("outputs", "loss_curve.png"))
plt.close()

plt.figure()
plt.plot(df_hist["train_acc"], label="train_acc")
plt.plot(df_hist["val_acc"], label="val_acc")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.title("accuracy_curve")
plt.savefig(os.path.join("outputs", "accuracy_curve.png"))
plt.close()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
loader = DataLoader(test_dataset, batch_size=25, shuffle=True)
data_iter = iter(loader)
images, labels = next(data_iter)
import torch
artifact = joblib.load(os.path.join("models", "mnist_artifact.pkl"))
state_dict_path = artifact["state_dict_path"]
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
state = torch.load(state_dict_path, map_location=device)
model.load_state_dict(state)
model.eval()
with torch.no_grad():
    outputs = model(images.to(device))
    probs = torch.softmax(outputs, dim=1).cpu().numpy()
    preds = outputs.argmax(dim=1).cpu().numpy()
fig, axes = plt.subplots(5,5, figsize=(8,8))
for i, ax in enumerate(axes.flatten()):
    img = images[i].numpy().squeeze()
    ax.imshow(img, cmap="gray")
    ax.set_title(f"t:{labels[i]} p:{preds[i]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join("outputs", "sample_predictions.png"))
plt.close()
print("visualizations saved")
