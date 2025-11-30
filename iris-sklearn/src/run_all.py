import subprocess
import sys

subprocess.check_call([sys.executable, "src/train.py"])
subprocess.check_call([sys.executable, "src/evaluate.py"])
subprocess.check_call([sys.executable, "src/visualize.py"])
