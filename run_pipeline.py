from __future__ import annotations

import argparse
import json

from mmrag import MultimodalRAGService


_SERVICE: MultimodalRAGService | None = None


def _get_service() -> MultimodalRAGService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = MultimodalRAGService()
    return _SERVICE


def run_mmrag(question: str, image_path: str | None = None, **_: object) -> dict:
    response = _get_service().ask(question=question, image_path=image_path)
    return response.model_dump()


def run_mmrag_multimodal(question: str, image_path: str | None = None, **_: object) -> dict:
    return run_mmrag(question=question, image_path=image_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the grounded multimodal RAG pipeline.")
    parser.add_argument("--question", required=True, help="Question to ask the system.")
    parser.add_argument("--image", help="Optional local image path.")
    args = parser.parse_args()

    result = run_mmrag(question=args.question, image_path=args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
