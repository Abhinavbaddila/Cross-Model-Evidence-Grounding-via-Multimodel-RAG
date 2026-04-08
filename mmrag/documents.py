from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import chromadb

from .config import Settings
from .corpus import matched_terms, tokenize
from .indexing import BGETextEncoder, _distance_to_similarity
from .schemas import ProofItem


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    file_hash: str
    file_name: str
    source_kind: str
    page_number: int | None
    text: str
    image_url: str | None


def _file_sha256(file_path: str) -> str:
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        chunks.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


class DocumentIndexer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = chromadb.PersistentClient(path=str(settings.doc_chroma_dir))
        self.collection = self.client.get_or_create_collection(
            settings.document_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.encoder: BGETextEncoder | None = None
        self.manifest = self._load_manifest()
        self.chunk_meta: dict[str, DocumentChunk] = {}
        self._hydrate_chunk_meta()

    def _load_manifest(self) -> dict[str, object]:
        if not self.settings.doc_manifest_path.exists():
            return {"files": {}}
        try:
            return json.loads(self.settings.doc_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"files": {}}

    def _save_manifest(self) -> None:
        self.settings.doc_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.doc_manifest_path.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    def _hydrate_chunk_meta(self) -> None:
        files = self.manifest.get("files", {})
        if not isinstance(files, dict):
            return
        for file_hash, payload in files.items():
            file_name = str(payload.get("file_name", file_hash))
            for chunk_payload in payload.get("chunks", []):
                chunk = DocumentChunk(
                    chunk_id=str(chunk_payload["chunk_id"]),
                    file_hash=file_hash,
                    file_name=file_name,
                    source_kind=str(chunk_payload.get("source_kind", "document-page")),
                    page_number=chunk_payload.get("page_number"),
                    text=str(chunk_payload.get("text", "")),
                    image_url=chunk_payload.get("image_url"),
                )
                self.chunk_meta[chunk.chunk_id] = chunk

    def _ensure_encoder(self) -> None:
        if self.encoder is None:
            self.encoder = BGETextEncoder(self.settings.text_embedding_model)

    @property
    def document_count(self) -> int:
        files = self.manifest.get("files", {})
        return len(files) if isinstance(files, dict) else 0

    def ingest(self, file_path: str) -> tuple[str, list[DocumentChunk]]:
        file_hash = _file_sha256(file_path)
        file_name = Path(file_path).name
        files = self.manifest.setdefault("files", {})
        existing = files.get(file_hash)
        if isinstance(existing, dict):
            chunk_ids = [str(item["chunk_id"]) for item in existing.get("chunks", [])]
            chunks = [self.chunk_meta[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunk_meta]
            if chunks:
                return file_hash, chunks

        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            chunks = self._ingest_pdf(file_path, file_hash)
        elif suffix in {".csv", ".xlsx", ".xls"}:
            chunks = self._ingest_table(file_path, file_hash)
        else:
            chunks = self._ingest_text(file_path, file_hash)

        self._ensure_encoder()
        assert self.encoder is not None
        embeddings = self.encoder.encode([chunk.text for chunk in chunks], is_query=False, batch_size=8)
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings.tolist(),
            metadatas=[
                {
                    "file_hash": chunk.file_hash,
                    "file_name": chunk.file_name,
                    "source_kind": chunk.source_kind,
                    "page_number": chunk.page_number if chunk.page_number is not None else -1,
                    "image_url": chunk.image_url or "",
                }
                for chunk in chunks
            ],
        )

        files[file_hash] = {
            "file_name": file_name,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source_kind": chunk.source_kind,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                    "image_url": chunk.image_url,
                }
                for chunk in chunks
            ],
        }
        for chunk in chunks:
            self.chunk_meta[chunk.chunk_id] = chunk
        self._save_manifest()
        return file_hash, chunks

    def _ingest_pdf(self, file_path: str, file_hash: str) -> list[DocumentChunk]:
        import fitz

        document = fitz.open(file_path)
        chunks: list[DocumentChunk] = []
        try:
            for page_index, page in enumerate(document, start=1):
                text = page.get_text("text")
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False)
                page_image_name = f"doc-{file_hash[:12]}-page-{page_index}.png"
                page_image_path = self.settings.generated_dir / page_image_name
                pixmap.save(page_image_path)
                page_image_url = f"/uploads/generated/{page_image_name}"

                for chunk_index, chunk_text in enumerate(_split_text(text), start=1):
                    chunk_id = f"doc-{file_hash[:12]}-page-{page_index}-chunk-{chunk_index}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            file_hash=file_hash,
                            file_name=Path(file_path).name,
                            source_kind="document-page",
                            page_number=page_index,
                            text=chunk_text,
                            image_url=page_image_url,
                        )
                    )
        finally:
            document.close()
        return chunks

    def _ingest_table(self, file_path: str, file_hash: str) -> list[DocumentChunk]:
        import pandas as pd

        chunks: list[DocumentChunk] = []
        suffix = Path(file_path).suffix.lower()
        if suffix == ".csv":
            dataframes = [("sheet1", pd.read_csv(file_path))]
        else:
            workbook = pd.read_excel(file_path, sheet_name=None)
            dataframes = list(workbook.items())

        for table_index, (sheet_name, dataframe) in enumerate(dataframes, start=1):
            csv_text = dataframe.to_csv(index=False)
            for chunk_index, chunk_text in enumerate(_split_text(csv_text, max_chars=1400, overlap=180), start=1):
                chunk_id = f"table-{file_hash[:12]}-{table_index}-chunk-{chunk_index}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        file_hash=file_hash,
                        file_name=Path(file_path).name,
                        source_kind="table-chunk",
                        page_number=table_index,
                        text=f"Table {sheet_name}\n{chunk_text}",
                        image_url=None,
                    )
                )
        return chunks

    def _ingest_text(self, file_path: str, file_hash: str) -> list[DocumentChunk]:
        raw_text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        chunks: list[DocumentChunk] = []
        for chunk_index, chunk_text in enumerate(_split_text(raw_text), start=1):
            chunk_id = f"text-{file_hash[:12]}-chunk-{chunk_index}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    file_hash=file_hash,
                    file_name=Path(file_path).name,
                    source_kind="text-chunk",
                    page_number=None,
                    text=chunk_text,
                    image_url=None,
                )
            )
        return chunks

    def search(self, file_path: str, question: str, top_k: int = 4) -> list[ProofItem]:
        file_hash = _file_sha256(file_path)
        file_name = Path(file_path).name
        files = self.manifest.setdefault("files", {})
        existing = files.get(file_hash)

        if isinstance(existing, dict):
            chunk_ids = [str(item["chunk_id"]) for item in existing.get("chunks", [])]
            chunks = [self.chunk_meta[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunk_meta]
        else:
            suffix = Path(file_path).suffix.lower()
            if suffix == ".pdf":
                chunks = self._ingest_pdf(file_path, file_hash)
            elif suffix in {".csv", ".xlsx", ".xls"}:
                chunks = self._ingest_table(file_path, file_hash)
            else:
                chunks = self._ingest_text(file_path, file_hash)

            files[file_hash] = {
                "file_name": file_name,
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "source_kind": chunk.source_kind,
                        "page_number": chunk.page_number,
                        "text": chunk.text,
                        "image_url": chunk.image_url,
                    }
                    for chunk in chunks
                ],
            }
            for chunk in chunks:
                self.chunk_meta[chunk.chunk_id] = chunk
            self._save_manifest()

        dense_proofs = self._search_dense_if_ready(file_hash, file_path, question, top_k)
        if dense_proofs:
            return dense_proofs
        return self._search_lexical(file_hash, file_name, chunks, question, top_k)

    def _search_dense_if_ready(self, file_hash: str, file_path: str, question: str, top_k: int) -> list[ProofItem]:
        if self.encoder is None:
            return []

        query_embedding = self.encoder.encode([question], is_query=True)
        result = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            where={"file_hash": file_hash},
            include=["documents", "metadatas", "distances"],
        )

        proofs: list[ProofItem] = []
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for index, (document, metadata, distance) in enumerate(zip(documents, metadatas, distances), start=1):
            if metadata is None:
                continue
            source_kind = str(metadata.get("source_kind", "document-page"))
            page_number = int(metadata.get("page_number", -1))
            matched = matched_terms(question, document or "")
            proofs.append(
                ProofItem(
                    id=f"D{index}",
                    title=f"{metadata.get('file_name', Path(file_path).name)}",
                    source_kind=source_kind if source_kind in {"document-page", "table-chunk", "text-chunk"} else "document-page",
                    source_id=str(metadata.get("file_hash", file_hash)),
                    page_number=None if page_number < 0 else page_number,
                    image_url=str(metadata.get("image_url") or "") or None,
                    caption=f"Retrieved from {metadata.get('file_name', Path(file_path).name)}",
                    supporting_text=document or "",
                    score=round(_distance_to_similarity(distance), 4),
                    retrieval_channels=["document_dense_text"],
                    matched_terms=matched,
                    explanation=[
                        "This proof was retrieved from the uploaded document after extracting and embedding its text.",
                        "The closest chunk was selected from the same uploaded file.",
                    ],
                    metadata={"file_hash": file_hash},
                )
            )
        return proofs

    def _search_lexical(
        self,
        file_hash: str,
        file_name: str,
        chunks: list[DocumentChunk],
        question: str,
        top_k: int,
    ) -> list[ProofItem]:
        query_tokens = set(tokenize(question))
        ranked: list[tuple[float, DocumentChunk, list[str]]] = []
        for index, chunk in enumerate(chunks):
            matches = matched_terms(question, chunk.text)
            overlap = len(matches)
            query_coverage = overlap / max(1, len(query_tokens))
            position_bonus = 1.0 / (index + 2.0)
            score = (0.65 * query_coverage) + (0.25 * min(1.0, overlap / 4.0)) + (0.10 * position_bonus)
            if overlap == 0 and question.strip():
                continue
            ranked.append((score, chunk, matches))

        if not ranked:
            ranked = [
                (1.0 / (index + 2.0), chunk, matched_terms(question, chunk.text))
                for index, chunk in enumerate(chunks[:top_k])
            ]

        ranked.sort(key=lambda item: item[0], reverse=True)
        proofs: list[ProofItem] = []
        for index, (score, chunk, matches) in enumerate(ranked[:top_k], start=1):
            proofs.append(
                ProofItem(
                    id=f"D{index}",
                    title=file_name,
                    source_kind=chunk.source_kind if chunk.source_kind in {"document-page", "table-chunk", "text-chunk"} else "document-page",
                    source_id=file_hash,
                    page_number=chunk.page_number,
                    image_url=chunk.image_url,
                    caption=f"Retrieved from {file_name}",
                    supporting_text=chunk.text,
                    score=round(min(0.999, 0.35 + score), 4),
                    retrieval_channels=["document_lexical"],
                    matched_terms=matches,
                    explanation=[
                        "This proof was selected directly from the uploaded document using lexical overlap on the extracted chunks.",
                        "Dense document indexing can be enabled later without changing the API.",
                    ],
                    metadata={"file_hash": file_hash},
                )
            )
        return proofs
