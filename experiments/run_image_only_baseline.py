import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

from src.retrieval.image_retrieval import ImageRetriever
from src.generation.answer_generator import AnswerGenerator
from src.generation.grounding_processor import GroundingProcessor
from src.evaluation.vqa_accuracy import VQAEvaluator


ir = ImageRetriever("indexes/image_index.faiss", "indexes/image_meta.json")
gen = AnswerGenerator()
ground = GroundingProcessor()
vqa_eval = VQAEvaluator()


def run_sample(question, gt_answers, top_k=3):
    image_results = ir.search(question, top_k=top_k)

    evidence = []
    for item in image_results:
        evidence.append({
            "image_id": item["image_id"],
            "image_path": item["image_path"],
            "caption": "",  # no captions in this baseline
            "score": item["score"]
        })

    answer = gen.generate(question, evidence)
    grounded = ground.attach(answer, evidence)
    score = vqa_eval.score(answer, gt_answers)

    return {"answer": answer, "grounded": grounded, "vqa": score}


if __name__ == "__main__":
    os.makedirs("experiments/model_outputs", exist_ok=True)

    sample = run_sample(
        "What is the man doing?",
        [{"answer": "playing tennis"}]
    )

    with open("experiments/model_outputs/image_only.json", "w") as f:
        json.dump(sample, f, indent=2)

    print("done image baseline")
