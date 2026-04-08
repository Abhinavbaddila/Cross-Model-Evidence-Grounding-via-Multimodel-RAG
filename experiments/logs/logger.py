import os
from datetime import datetime

LOG_DIR = "experiments/logs/"

os.makedirs(LOG_DIR, exist_ok=True)

def log(message, filename="run.log"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(LOG_DIR, filename)
    with open(path, "a") as f:
        f.write(f"[{ts}] {message}\n")
