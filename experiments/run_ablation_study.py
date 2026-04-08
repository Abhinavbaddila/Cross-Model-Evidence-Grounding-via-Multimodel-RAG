import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_pipeline import run_mmrag
import json
from datetime import datetime
from tqdm import tqdm

ABLATIONS = {
    "full": {"text": True, "image": True},
    "text_only": {"text": True, "image": False},
    "image_only": {"text": False, "image": True},
}

QUESTIONS = [
    ("What is the man doing?", [{"answer": "playing tennis"}]),
    ("What animal is shown?", [{"answer": "dog"}]),
    ("What vehicle is present?", [{"answer": "car"}]),
]

results = []

for mode, cfg in ABLATIONS.items():
    print(f"\nRunning mode: {mode}")

    for q, gt in tqdm(QUESTIONS):
        out = run_mmrag(
            question=q,
            gt_answers=gt,
            split="train"
        )
        out["mode"] = mode
        results.append(out)

os.makedirs("experiments/ablations", exist_ok=True)

with open("experiments/ablations/ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Ablation study completed!")
