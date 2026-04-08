from __future__ import annotations

import contextlib
import io
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
)

from .config import Settings
from .corpus import (
    CorpusRecord,
    TextChunk,
    build_coco_val_corpus,
    compute_image_dhash,
    compute_image_sha256,
    hamming_distance_hex,
    read_corpus_cache,
    salient_terms,
    tokenize,
    write_corpus_cache,
)


def _normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return array / norms


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lower = min(values)
    upper = max(values)
    if abs(upper - lower) < 1e-9:
        return [1.0 for _ in values]
    return [(value - lower) / (upper - lower) for value in values]


def _distance_to_similarity(distance: float) -> float:
    return max(0.0, 1.0 - float(distance))


def _model_output_to_tensor(output):
    return output.pooler_output if hasattr(output, "pooler_output") else output


@contextlib.contextmanager
def _suppress_model_load_output():
    from huggingface_hub.utils import logging as hf_logging
    from transformers import logging as transformers_logging

    previous_hf_level = hf_logging.get_verbosity()
    previous_level = transformers_logging.get_verbosity()
    sink = io.StringIO()
    hf_logging.set_verbosity_error()
    transformers_logging.set_verbosity_error()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
    finally:
        hf_logging.set_verbosity(previous_hf_level)
        transformers_logging.set_verbosity(previous_level)


class BGETextEncoder:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with _suppress_model_load_output():
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str], *, is_query: bool, batch_size: int = 12) -> np.ndarray:
        prepared = texts
        if is_query:
            prepared = [f"Represent this sentence for searching relevant passages: {text}" for text in texts]

        batches: list[np.ndarray] = []
        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start : start + batch_size]
            inputs = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            pooled = outputs.last_hidden_state[:, 0]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(pooled.cpu().numpy().astype("float32"))

        return np.vstack(batches) if batches else np.empty((0, 1024), dtype=np.float32)


class CLIPEmbedder:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with _suppress_model_load_output():
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model.get_text_features(**inputs)
        features = _model_output_to_tensor(outputs)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy().astype("float32")

    @torch.no_grad()
    def encode_images(self, image_paths: list[str], batch_size: int = 8) -> np.ndarray:
        from PIL import Image

        batches: list[np.ndarray] = []
        for start in range(0, len(image_paths), batch_size):
            chunk = image_paths[start : start + batch_size]
            images = [Image.open(path).convert("RGB") for path in chunk]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            features = _model_output_to_tensor(outputs)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            batches.append(features.cpu().numpy().astype("float32"))
            for image in images:
                image.close()
        return np.vstack(batches) if batches else np.empty((0, 512), dtype=np.float32)


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        with _suppress_model_load_output():
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, query: str, documents: list[str], batch_size: int = 16) -> list[float]:
        if not documents:
            return []

        scores: list[float] = []
        for start in range(0, len(documents), batch_size):
            chunk = documents[start : start + batch_size]
            pairs = [(query, document) for document in chunk]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**inputs).logits.squeeze(-1)
            scores.extend(float(value) for value in logits.detach().cpu().tolist())
        return scores


@dataclass(slots=True)
class RetrievalHit:
    doc_id: str
    score: float
    channel: str
    chunk_id: str | None = None


@dataclass(slots=True)
class PreparedAssets:
    corpus: list[CorpusRecord]
    image_index_source: str
    vector_store: str


class CorpusIndexer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.corpus: list[CorpusRecord] = []
        self.text_chunks: list[TextChunk] = []
        self.corpus_by_id: dict[str, CorpusRecord] = {}
        self.corpus_by_image_path: dict[str, CorpusRecord] = {}
        self.corpus_by_sha256: dict[str, CorpusRecord] = {}
        self.corpus_dhash_pairs: list[tuple[str, CorpusRecord]] = []
        self.chunk_by_id: dict[str, TextChunk] = {}
        self.chunks_by_doc_id: dict[str, list[TextChunk]] = {}

        self.text_encoder: BGETextEncoder | None = None
        self.clip: CLIPEmbedder | None = None
        self.reranker: CrossEncoderReranker | None = None

        self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.text_collection = None
        self.image_collection = None
        self.image_index_source = "chromadb-missing"
        self.vector_store = "chromadb"

        self.bm25: BM25Okapi | None = None
        self._corpus_ready = False
        self._is_ready = False
        self._estimated_document_count = 0
        self._state_lock = threading.RLock()
        self._prepare_lock = threading.Lock()

    @property
    def active_models(self) -> dict[str, str]:
        return {
            "text_embedding": self.text_encoder.model_name if self.text_encoder is not None else self.settings.text_embedding_model,
            "image_embedding": self.clip.model_name if self.clip is not None else self.settings.image_embedding_model,
            "reranker": self.reranker.model_name if self.reranker is not None else ("disabled" if not self.settings.use_reranker else self.settings.reranker_model),
        }

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def corpus_ready(self) -> bool:
        return self._corpus_ready

    def ensure_corpus_ready(self) -> PreparedAssets:
        if self._corpus_ready:
            return PreparedAssets(corpus=self.corpus, image_index_source=self.image_index_source, vector_store=self.vector_store)

        with self._state_lock:
            if self._corpus_ready:
                return PreparedAssets(corpus=self.corpus, image_index_source=self.image_index_source, vector_store=self.vector_store)

            self.corpus = self._load_or_build_corpus()
            self._estimated_document_count = len(self.corpus)
            if not self.corpus:
                raise RuntimeError(
                    "No corpus records were loaded. Check the captions file, image directory, and cached corpus data."
                )
            self.text_chunks = [chunk for record in self.corpus for chunk in record.chunks]
            if not self.text_chunks:
                raise RuntimeError(
                    "The corpus loaded successfully but no text chunks were available. Rebuild the corpus cache."
                )
            self.corpus_by_id = {record.doc_id: record for record in self.corpus}
            self.corpus_by_image_path = {self._normalize_image_path(record.image_path): record for record in self.corpus}
            self.corpus_by_sha256 = {record.image_sha256: record for record in self.corpus if record.image_sha256}
            self.corpus_dhash_pairs = [(record.image_dhash, record) for record in self.corpus if record.image_dhash]
            self.chunk_by_id = {chunk.chunk_id: chunk for chunk in self.text_chunks}
            self.chunks_by_doc_id = {record.doc_id: record.chunks for record in self.corpus}
            self.bm25 = BM25Okapi([tokenize(chunk.chunk_text) for chunk in self.text_chunks])
            self._corpus_ready = True
        return PreparedAssets(corpus=self.corpus, image_index_source=self.image_index_source, vector_store=self.vector_store)

    def prepare(self) -> PreparedAssets:
        if self._is_ready:
            return PreparedAssets(corpus=self.corpus, image_index_source=self.image_index_source, vector_store=self.vector_store)

        self.ensure_corpus_ready()

        with self._prepare_lock:
            if self._is_ready:
                return PreparedAssets(corpus=self.corpus, image_index_source=self.image_index_source, vector_store=self.vector_store)

            self._ensure_models()
            self._load_or_build_collections()
            with self._state_lock:
                self._is_ready = True
        return PreparedAssets(corpus=self.corpus, image_index_source=self.image_index_source, vector_store=self.vector_store)

    def estimated_document_count(self) -> int:
        if self.corpus:
            return len(self.corpus)
        if self._estimated_document_count:
            return self._estimated_document_count

        manifest = self._read_manifest()
        if manifest and isinstance(manifest.get("image_count"), int):
            self._estimated_document_count = int(manifest["image_count"])
            return self._estimated_document_count

        if self.settings.corpus_cache_path.exists():
            try:
                self._estimated_document_count = len(read_corpus_cache(self.settings.corpus_cache_path))
                return self._estimated_document_count
            except (json.JSONDecodeError, TypeError, OSError):
                return 0

        return 0

    def _ensure_models(self) -> None:
        if self.text_encoder is None:
            self.text_encoder = BGETextEncoder(self.settings.text_embedding_model)
        if self.clip is None:
            self.clip = CLIPEmbedder(self.settings.image_embedding_model)
        if self.settings.use_reranker and self.reranker is None:
            self.reranker = CrossEncoderReranker(self.settings.reranker_model)

    def _load_or_build_corpus(self) -> list[CorpusRecord]:
        if self.settings.corpus_cache_path.exists():
            try:
                records = read_corpus_cache(self.settings.corpus_cache_path)
                if records and all(record.chunks for record in records):
                    return records
            except (json.JSONDecodeError, TypeError):
                pass

        records = build_coco_val_corpus(self.settings.captions_path, self.settings.val_images_dir)
        write_corpus_cache(records, self.settings.corpus_cache_path)
        return records

    def _load_or_build_collections(self) -> None:
        if self._collections_are_current():
            self.text_collection = self.client.get_collection(self.settings.text_collection_name)
            self.image_collection = self.client.get_collection(self.settings.image_collection_name)
            self.image_index_source = "chromadb-cache"
            return

        self._reset_collection(self.settings.text_collection_name)
        self._reset_collection(self.settings.image_collection_name)
        self.text_collection = self.client.get_or_create_collection(
            self.settings.text_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.image_collection = self.client.get_or_create_collection(
            self.settings.image_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._index_text_chunks()
        self._index_images()
        self._write_manifest()
        self.image_index_source = "chromadb-rebuilt"

    def _collections_are_current(self) -> bool:
        manifest = self._read_manifest()
        if not manifest:
            return False

        expected = {
            "text_model": self.settings.text_embedding_model,
            "image_model": self.settings.image_embedding_model,
            "chunk_count": len(self.text_chunks),
            "image_count": len(self.corpus),
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            return False

        try:
            text_collection = self.client.get_collection(self.settings.text_collection_name)
            image_collection = self.client.get_collection(self.settings.image_collection_name)
        except Exception:
            return False

        return text_collection.count() == len(self.text_chunks) and image_collection.count() == len(self.corpus)

    def _reset_collection(self, name: str) -> None:
        try:
            self.client.delete_collection(name)
        except Exception:
            pass

    def _write_manifest(self) -> None:
        payload = {
            "text_model": self.settings.text_embedding_model,
            "image_model": self.settings.image_embedding_model,
            "chunk_count": len(self.text_chunks),
            "image_count": len(self.corpus),
        }
        self.settings.chroma_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _read_manifest(self) -> dict[str, object] | None:
        if not self.settings.chroma_manifest_path.exists():
            return None
        try:
            return json.loads(self.settings.chroma_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _index_text_chunks(self) -> None:
        assert self.text_encoder is not None
        ids = [chunk.chunk_id for chunk in self.text_chunks]
        documents = [chunk.chunk_text for chunk in self.text_chunks]
        metadatas = [
            {
                "doc_id": chunk.doc_id,
                "image_id": chunk.image_id,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "image_url": chunk.image_url,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            }
            for chunk in self.text_chunks
        ]
        embeddings = self.text_encoder.encode(
            documents,
            is_query=False,
            batch_size=max(1, min(self.settings.embedding_batch_size, 12)),
        )
        self._batch_upsert(self.text_collection, ids, documents, metadatas, embeddings)

    def _index_images(self) -> None:
        assert self.clip is not None
        ids = [record.doc_id for record in self.corpus]
        documents = [record.merged_text for record in self.corpus]
        metadatas = [
            {
                "doc_id": record.doc_id,
                "image_id": record.image_id,
                "image_url": record.image_url,
                "lead_chunk_id": record.chunks[0].chunk_id,
            }
            for record in self.corpus
        ]
        embeddings = self.clip.encode_images(
            [record.image_path for record in self.corpus],
            batch_size=max(1, min(self.settings.embedding_batch_size, 8)),
        )
        self._batch_upsert(self.image_collection, ids, documents, metadatas, embeddings)

    def _batch_upsert(self, collection, ids: list[str], documents: list[str], metadatas: list[dict[str, object]], embeddings: np.ndarray) -> None:
        batch_size = max(1, min(self.settings.embedding_batch_size, 64))
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end].tolist(),
            )

    def _normalize_image_path(self, raw_path: str) -> str:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.settings.project_root / path
        return str(path.resolve())

    def search_dense_text(self, query: str, top_k: int) -> list[RetrievalHit]:
        assert self.text_encoder is not None
        assert self.text_collection is not None
        query_embedding = self.text_encoder.encode([query], is_query=True)
        result = self.text_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        return self._chunk_query_hits(result, "dense_text")

    def search_bm25(self, query: str, top_k: int) -> list[RetrievalHit]:
        assert self.bm25 is not None
        scores = self.bm25.get_scores(tokenize(query))
        ranked = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievalHit(
                doc_id=self.text_chunks[index].doc_id,
                score=float(scores[index]),
                channel="bm25",
                chunk_id=self.text_chunks[index].chunk_id,
            )
            for index in ranked
            if scores[index] > 0
        ]

    def search_clip_text(self, query: str, top_k: int) -> list[RetrievalHit]:
        assert self.clip is not None
        assert self.image_collection is not None
        query_embedding = self.clip.encode_text([query])
        result = self.image_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        return self._image_query_hits(result, "clip_text")

    def search_clip_image(self, image_path: str, top_k: int) -> list[RetrievalHit]:
        assert self.clip is not None
        assert self.image_collection is not None
        query_embedding = self.clip.encode_images([image_path])
        result = self.image_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        return self._image_query_hits(result, "clip_image")

    def _chunk_query_hits(self, result: dict[str, list], channel: str) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for metadata, distance in zip(result.get("metadatas", [[]])[0], result.get("distances", [[]])[0]):
            if metadata is None:
                continue
            hits.append(
                RetrievalHit(
                    doc_id=str(metadata["doc_id"]),
                    score=_distance_to_similarity(distance),
                    channel=channel,
                    chunk_id=str(metadata["chunk_id"]),
                )
            )
        return hits

    def _image_query_hits(self, result: dict[str, list], channel: str) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for metadata, distance in zip(result.get("metadatas", [[]])[0], result.get("distances", [[]])[0]):
            if metadata is None:
                continue
            hits.append(
                RetrievalHit(
                    doc_id=str(metadata["doc_id"]),
                    score=_distance_to_similarity(distance),
                    channel=channel,
                    chunk_id=str(metadata.get("lead_chunk_id", "")) or None,
                )
            )
        return hits

    def find_exact_image_match(self, image_path: str) -> RetrievalHit | None:
        normalized_path = self._normalize_image_path(image_path)
        record = self.corpus_by_image_path.get(normalized_path)
        if record is not None:
            return RetrievalHit(
                doc_id=record.doc_id,
                score=1.0,
                channel="exact_image_match",
                chunk_id=record.chunks[0].chunk_id,
            )

        try:
            image_sha256 = compute_image_sha256(Path(image_path))
        except OSError:
            return None

        record = self.corpus_by_sha256.get(image_sha256)
        if record is None:
            return None
        return RetrievalHit(
            doc_id=record.doc_id,
            score=1.0,
            channel="exact_image_match",
            chunk_id=record.chunks[0].chunk_id,
        )

    def search_near_image_matches(self, image_path: str, top_k: int, max_distance: int = 8) -> list[RetrievalHit]:
        exact_match = self.find_exact_image_match(image_path)
        if exact_match is not None:
            return []

        try:
            query_hash = compute_image_dhash(Path(image_path))
        except OSError:
            return []

        candidates: list[tuple[int, float, str]] = []
        for corpus_hash, record in self.corpus_dhash_pairs:
            distance = hamming_distance_hex(query_hash, corpus_hash)
            if distance > max_distance:
                continue
            score = 1.0 - (distance / 64.0)
            candidates.append((distance, score, record.doc_id))

        candidates.sort(key=lambda item: (item[0], -item[1], item[2]))
        return [
            RetrievalHit(
                doc_id=doc_id,
                score=score,
                channel="near_image_match",
                chunk_id=self.corpus_by_id[doc_id].chunks[0].chunk_id,
            )
            for _, score, doc_id in candidates[:top_k]
        ]

    def resolve_support_chunk(self, doc_id: str, query: str) -> TextChunk:
        chunks = self.chunks_by_doc_id[doc_id]
        query_terms = salient_terms(query)
        if not query_terms:
            return chunks[0]

        def chunk_score(chunk: TextChunk) -> tuple[int, int]:
            overlap = len(query_terms.intersection(salient_terms(chunk.chunk_text)))
            return overlap, -chunk.chunk_index

        return max(chunks, key=chunk_score)

    def get_chunk(self, chunk_id: str) -> TextChunk:
        return self.chunk_by_id[chunk_id]

    def rerank(self, query: str, chunks: list[TextChunk]) -> dict[str, float]:
        if self.reranker is None:
            return {chunk.chunk_id: 0.0 for chunk in chunks}
        documents = [chunk.chunk_text for chunk in chunks]
        scores = self.reranker.score(query, documents)
        normalized = _minmax(scores)
        return {chunk.chunk_id: score for chunk, score in zip(chunks, normalized)}
