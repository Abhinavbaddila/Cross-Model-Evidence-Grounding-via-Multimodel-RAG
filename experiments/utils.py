import os
import json
from datetime import datetime

def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_log(log, name="log.txt"):
    path = f"experiments/logs/{timestamp()}_{name}"
    with open(path, "w") as f:
        f.write(log)
