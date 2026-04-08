from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_root: Path
    captions_path: Path
    val_images_dir: Path
    upload_dir: Path
    index_dir: Path
    generated_dir: Path
    corpus_cache_path: Path
    chroma_dir: Path
    chroma_manifest_path: Path
    documents_dir: Path
    doc_chroma_dir: Path
    doc_manifest_path: Path
    text_collection_name: str
    image_collection_name: str
    document_collection_name: str
    top_k: int
    query_fusion_k: int
    embedding_batch_size: int
    visual_proof_count: int
    detection_threshold: float
    text_embedding_model: str
    image_embedding_model: str
    reranker_model: str
    use_reranker: bool
    vlm_provider: str
    vlm_model: str
    caption_model: str
    vqa_model: str
    detector_provider: str
    detector_model: str
    detector_fallback_model: str
    llm_provider: str
    llama_model: str
    openai_api_key: str | None
    openai_model: str
    use_openai_if_available: bool

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.doc_chroma_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parent.parent
    data_root = Path(os.getenv("MMRAG_DATA_ROOT", str(project_root / "data" / "coco")))
    index_dir = Path(os.getenv("MMRAG_INDEX_DIR", str(project_root / "indexes" / "latest_mmrag")))
    chroma_dir = index_dir / "chromadb"
    documents_dir = index_dir / "documents"
    doc_chroma_dir = documents_dir / "chromadb"
    generated_dir = Path(os.getenv("MMRAG_GENERATED_DIR", str(project_root / "uploads" / "generated")))

    settings = Settings(
        project_root=project_root,
        data_root=data_root,
        captions_path=Path(os.getenv("MMRAG_CAPTIONS_PATH", str(data_root / "captions_val2017.json"))),
        val_images_dir=Path(os.getenv("MMRAG_VAL_IMAGES_DIR", str(data_root / "val2017"))),
        upload_dir=Path(os.getenv("MMRAG_UPLOAD_DIR", str(project_root / "uploads"))),
        index_dir=index_dir,
        generated_dir=generated_dir,
        corpus_cache_path=index_dir / "corpus.json",
        chroma_dir=chroma_dir,
        chroma_manifest_path=index_dir / "chroma_manifest.json",
        documents_dir=documents_dir,
        doc_chroma_dir=doc_chroma_dir,
        doc_manifest_path=documents_dir / "document_manifest.json",
        text_collection_name=os.getenv("MMRAG_TEXT_COLLECTION", "mmrag_text_chunks"),
        image_collection_name=os.getenv("MMRAG_IMAGE_COLLECTION", "mmrag_image_embeddings"),
        document_collection_name=os.getenv("MMRAG_DOCUMENT_COLLECTION", "mmrag_document_chunks"),
        top_k=int(os.getenv("MMRAG_TOP_K", "4")),
        query_fusion_k=int(os.getenv("MMRAG_FUSION_K", "8")),
        embedding_batch_size=int(os.getenv("MMRAG_EMBED_BATCH", "8")),
        visual_proof_count=int(os.getenv("MMRAG_VISUAL_PROOFS", "4")),
        detection_threshold=float(os.getenv("MMRAG_DETECTION_THRESHOLD", "0.22")),
        text_embedding_model=os.getenv("MMRAG_TEXT_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
        image_embedding_model=os.getenv("MMRAG_IMAGE_EMBEDDING_MODEL", "openai/clip-vit-base-patch32"),
        reranker_model=os.getenv("MMRAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        use_reranker=os.getenv("MMRAG_USE_RERANKER", "1") not in {"0", "false", "False"},
        vlm_provider=os.getenv("MMRAG_VLM_PROVIDER", "auto").strip().lower(),
        vlm_model=os.getenv("MMRAG_VLM_MODEL", "llava-hf/llava-v1.6-mistral-7b-hf"),
        caption_model=os.getenv("MMRAG_CAPTION_MODEL", "Salesforce/blip-image-captioning-base"),
        vqa_model=os.getenv("MMRAG_VQA_MODEL", "Salesforce/blip-vqa-base"),
        detector_provider=os.getenv("MMRAG_DETECTOR_PROVIDER", "yolo").strip().lower(),
        detector_model=os.getenv("MMRAG_DETECTOR_MODEL", "yolov8n.pt"),
        detector_fallback_model=os.getenv("MMRAG_DETECTOR_FALLBACK_MODEL", "google/owlv2-base-patch16-ensemble"),
        llm_provider=os.getenv("MMRAG_LLM_PROVIDER", "extractive").strip().lower(),
        llama_model=os.getenv("MMRAG_LLAMA_MODEL", "meta-llama/Llama-3.2-3B-Instruct"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        use_openai_if_available=os.getenv("MMRAG_USE_OPENAI", "1") not in {"0", "false", "False"},
    )
    settings.ensure_dirs()
    return settings
