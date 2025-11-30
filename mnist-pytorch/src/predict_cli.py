import os
import sys
import joblib
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

MODEL_ARTIFACT = os.path.join("models", "mnist_artifact.pkl")
if not os.path.exists(MODEL_ARTIFACT):
    print("model_missing")
    sys.exit(1)

artifact = joblib.load(MODEL_ARTIFACT)
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

transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((28,28)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

def predict_from_image_path(path):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(out.argmax(dim=1).cpu().numpy()[0])
    return pred, probs

print("mnist_cli_ready")
while True:
    s = input("type 'sample <index>' to predict test sample, 'path <imagepath>' to predict an image, or 'exit': ").strip()
    if s.lower() == "exit":
        break
    if s.startswith("sample"):
        parts = s.split()
        if len(parts) != 2:
            print("invalid")
            continue
        try:
            idx = int(parts[1])
        except:
            print("invalid_index")
            continue
        from torchvision import datasets
        import torch
        transform_simple = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        test = datasets.MNIST(root="data", train=False, download=True, transform=transform_simple)
        img, label = test[idx]
        x = img.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred = int(out.argmax(dim=1).cpu().numpy()[0])
        print("true_label", int(label))
        print("predicted", pred)
        print("probabilities", {i: float(probs[i]) for i in range(10)})
    elif s.startswith("path"):
        parts = s.split(maxsplit=1)
        if len(parts) != 2:
            print("invalid")
            continue
        path = parts[1].strip()
        if not os.path.exists(path):
            print("file_not_found")
            continue
        pred, probs = predict_from_image_path(path)
        print("predicted", pred)
        print("probabilities", {i: float(probs[i]) for i in range(10)})
    else:
        print("unknown_command")
print("bye")
