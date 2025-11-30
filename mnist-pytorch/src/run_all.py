import argparse
import subprocess
import sys
import os
parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true")
args = parser.parse_args()
model_path = os.path.join("models", "mnist_cnn.pt")
if args.force or (not os.path.exists(model_path)):
    subprocess.check_call([sys.executable, "src/train.py"])
else:
    print(f"model found at {model_path}, skipping training (use --force to retrain)")
subprocess.check_call([sys.executable, "src/evaluate.py"])
subprocess.check_call([sys.executable, "src/visualize.py"])
