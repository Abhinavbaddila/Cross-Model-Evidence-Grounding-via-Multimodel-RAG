import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_pipeline import run_mmrag
import json
from tqdm import tqdm
from datetime import datetime

# small evaluation set
QUESTIONS = [
    ("What is the man doing?", [{"answer": "playing tennis"}]),
    ("What is the woman holding?", [{"answer": "umbrella"}]),
    ("What vehicle is present?", [{"answer": "car"}]),
    ("What animal do you see?", [{"answer": "dog"}]),
]

results = []

for q, gt in tqdm(QUESTIONS):
    out = run_mmrag(q, gt, split="train")
    results.append(out)

os.makedirs("experiments/results", exist_ok=True)

with open("experiments/results/batch_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Batch evaluation complete!")
