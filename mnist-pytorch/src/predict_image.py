import sys
import os
from PIL import Image
import joblib
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if len(sys.argv) < 2:
    print("usage: python src/predict_image.py path/to/image.png")
    sys.exit(1)

path = sys.argv[1]
if not os.path.exists(path):
    print("file_not_found")
    sys.exit(1)

os.makedirs("outputs", exist_ok=True)

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

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

img = Image.open(path).convert("RGB")
processed = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    out = model(processed)
    probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    pred = int(out.argmax(dim=1).cpu().numpy()[0])

processed_img = processed.cpu().numpy()[0][0]

plt.figure(figsize=(3,3))
plt.imshow(processed_img, cmap="gray")
plt.axis("off")
processed_path = os.path.join("outputs", "processed_image.png")
plt.savefig(processed_path, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6,4))
plt.bar(np.arange(10), probs)
plt.xticks(np.arange(10))
plt.xlabel("digit")
plt.ylabel("probability")
plt.title("prediction_probabilities")
bar_path = os.path.join("outputs", "probabilities.png")
plt.savefig(bar_path)
plt.close()

df = pd.DataFrame({
    "digit": list(range(10)),
    "probability": probs
})
df["predicted_digit"] = pred
csv_path = os.path.join("outputs", "prediction_output.csv")
df.to_csv(csv_path, index=False)

print("predicted", pred)
print("probabilities", {i: float(probs[i]) for i in range(10)})
print("saved", processed_path)
print("saved", bar_path)
print("saved", csv_path)
