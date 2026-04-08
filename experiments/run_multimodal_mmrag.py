import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from datetime import datetime

from src.retrieval.text_retrieval import TextRetriever
from src.retrieval.image_retrieval import ImageRetriever
from src.fusion.reranker import EvidenceReranker
from src.fusion.fuse_evidence import EvidenceFuser
from src.generation.answer_generator import AnswerGenerator
from src.generation.grounding_processor import GroundingProcessor
from src.evaluation.vqa_accuracy import VQAEvaluator
from src.evaluation.faithfulness import FaithfulnessEvaluator


# modules
tr = TextRetriever("indexes/text_index.faiss", "indexes/text_meta.json")
ir = ImageRetriever("indexes/image_index.faiss", "indexes/image_meta.json")
rerank = EvidenceReranker()
fuser = EvidenceFuser(coco_base="data/coco")
gen = AnswerGenerator()
ground = GroundingProcessor()
vqa_eval = VQAEvaluator()
faith_eval = FaithfulnessEvaluator()


def run_mmrag(question, gt_answers, split="val", top_k=3):
    text_results = tr.search(question, top_k=top_k)
    image_results = ir.search(question, top_k=top_k)

    fused = rerank.combine(text_results, image_results)
    structured = fuser.fuse(question, fused, split=split, top_k=top_k)

    answer = gen.generate(question, structured["evidence"])
    grounded = ground.attach(answer, structured["evidence"])

    vqa_score = vqa_eval.score(answer, gt_answers)
    faith_score = faith_eval.score(answer, grounded["sources"])

    return {
        "question": question,
        "answer": answer,
        "grounded": grounded,
        "timestamp": str(datetime.now()),
        "vqa": vqa_score,
        "faith": faith_score
    }


if __name__ == "__main__":
    os.makedirs("experiments/model_outputs", exist_ok=True)

    sample = run_mmrag(
        "What is the man doing?",
        [{"answer": "playing tennis"}],
        split="train"
    )

    with open("experiments/model_outputs/mmrag.json", "w") as f:
        json.dump(sample, f, indent=2)

    print("done multimodal mmrag")
