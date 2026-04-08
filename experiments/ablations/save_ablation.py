import json
import os
from datetime import datetime

ABL_DIR = "experiments/ablations/"
os.makedirs(ABL_DIR, exist_ok=True)

def save_ablation(results, name="ablation"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(ABL_DIR, f"{name}_{ts}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path
