import json
from experiments.run_multimodal_mmrag import run_sample as mmrag_run
from experiments.run_text_only_baseline import run_sample as text_run
from experiments.run_image_only_baseline import run_sample as image_run

ablation_cases = [
    ("text_only", text_run),
    ("image_only", image_run),
    ("mmrag", mmrag_run),
]

results = {}

for name, fn in ablation_cases:
    out = fn("What is the man doing?", [{"answer": "playing tennis"}])
    results[name] = out

with open("experiments/ablations/ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("done ablations")
