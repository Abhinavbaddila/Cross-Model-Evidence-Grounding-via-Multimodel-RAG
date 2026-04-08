from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mmrag.config import get_settings


settings = get_settings()
_service = None
_service_lock = threading.Lock()

app = FastAPI(title="Hybrid Multimodal RAG", version="2.0.0")
app.state.startup_state = "cold"
app.state.startup_error = None
app.state.warmup_task = None
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(settings.upload_dir)), name="uploads")
app.mount("/assets/coco/val2017", StaticFiles(directory=str(settings.val_images_dir)), name="coco-val")


def _get_service():
    global _service
    if _service is not None:
        return _service

    with _service_lock:
        if _service is None:
            from mmrag import MultimodalRAGService

            _service = MultimodalRAGService(settings)
    return _service


def _estimate_document_count() -> int:
    cache_path = settings.corpus_cache_path
    if not cache_path.exists():
        return 0
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    return len(payload) if isinstance(payload, list) else 0


def _placeholder_status() -> dict[str, object]:
    return {
        "status": "ok",
        "corpus_name": "Hybrid Multimodal Corpus",
        "document_count": _estimate_document_count(),
        "document_store_count": 0,
        "vector_store": "chromadb",
        "image_index_source": "chromadb-pending",
        "generator_mode": "initializing",
        "active_models": {
            "text_embedding": settings.text_embedding_model,
            "image_embedding": settings.image_embedding_model,
            "reranker": settings.reranker_model if settings.use_reranker else "disabled",
            "generator": settings.openai_model if settings.llm_provider == "openai" else "extractive-fallback",
            "vlm": settings.vlm_model,
            "detector": settings.detector_model,
        },
        "capabilities": [
            "zero-shot image reasoning",
            "visual grounding",
            "similar image retrieval",
            "pdf ingestion",
            "table ingestion",
            "confidence scoring",
        ],
        "supported_file_types": [".jpg", ".jpeg", ".png", ".webp", ".pdf", ".csv", ".xlsx", ".xls", ".txt"],
        "ready": False,
        "startup_state": app.state.startup_state,
        "startup_error": app.state.startup_error,
    }


def _warmup_service() -> None:
    service = _get_service()
    service.warmup()


async def _background_warmup() -> None:
    try:
        await asyncio.to_thread(_warmup_service)
        app.state.startup_state = "ready"
        app.state.startup_error = None
    except Exception as exc:
        app.state.startup_state = "error"
        app.state.startup_error = str(exc)


@app.on_event("startup")
async def startup_event() -> None:
    app.state.startup_state = "warming"
    app.state.startup_error = None
    app.state.warmup_task = asyncio.create_task(_background_warmup())


@app.get("/api/health")
async def health() -> dict[str, str | None]:
    return {"status": "ok", "startup_state": app.state.startup_state, "startup_error": app.state.startup_error}


@app.get("/api/status")
async def status():
    if _service is None:
        return _placeholder_status()

    service = _get_service()
    return await asyncio.to_thread(
        service.status,
        startup_state=app.state.startup_state,
        startup_error=app.state.startup_error,
    )


@app.post("/api/query")
async def query(question: str = Form(""), file: UploadFile | None = File(default=None)):
    saved_path = None
    if file is not None and file.filename:
        extension = Path(file.filename).suffix or ".jpg"
        safe_name = f"{uuid.uuid4().hex}{extension.lower()}"
        saved_path = settings.upload_dir / safe_name
        content = await file.read()
        saved_path.write_bytes(content)

    try:
        service = await asyncio.to_thread(_get_service)
        response = await asyncio.to_thread(service.ask, question, str(saved_path) if saved_path else None)
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hybrid multimodal RAG backend is running."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webapp.app:app", host="0.0.0.0", port=int(os.getenv("PORT", "9000")), reload=True)
