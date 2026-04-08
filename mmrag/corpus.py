from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "with",
}


@dataclass(slots=True)
class TextChunk:
    chunk_id: str
    doc_id: str
    image_id: int
    image_url: str
    chunk_index: int
    chunk_text: str
    char_start: int
    char_end: int


@dataclass(slots=True)
class CorpusRecord:
    doc_id: str
    image_id: int
    image_path: str
    image_url: str
    image_sha256: str
    image_dhash: str
    title: str
    merged_text: str
    captions: list[str]
    keywords: list[str]
    chunks: list[TextChunk]


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def salient_terms(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS and len(token) > 2}


def _keywords_from_captions(captions: list[str], limit: int = 6) -> list[str]:
    counts = Counter()
    for caption in captions:
        for token in salient_terms(caption):
            counts[token] += 1
    return [token for token, _ in counts.most_common(limit)]


def compute_image_sha256(image_path: Path) -> str:
    digest = hashlib.sha256()
    with image_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_image_dhash(image_path: Path, hash_size: int = 8) -> str:
    from PIL import Image

    with Image.open(image_path) as image:
        grayscale = image.convert("L").resize((hash_size + 1, hash_size))
        pixels = np.asarray(grayscale, dtype=np.int16)

    differences = pixels[:, 1:] > pixels[:, :-1]
    bits = 0
    for value in differences.flatten():
        bits = (bits << 1) | int(value)

    width = (hash_size * hash_size) // 4
    return f"{bits:0{width}x}"


def hamming_distance_hex(left: str, right: str) -> int:
    if not left or not right:
        return 64
    return int((int(left, 16) ^ int(right, 16)).bit_count())


def matched_terms(query: str, chunk_text: str) -> list[str]:
    return sorted(salient_terms(query).intersection(salient_terms(chunk_text)))


def highlight_terms(text: str, query: str) -> tuple[str, list[str], list[tuple[int, int]]]:
    terms = matched_terms(query, text)
    if not terms:
        return html.escape(text), [], []

    term_set = {term.lower() for term in terms}
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0

    # Highlight only full token matches so proof spans remain precise and traceable.
    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        pieces.append(html.escape(text[cursor : match.start()]))
        if token.lower() in term_set:
            pieces.append(f"<mark>{html.escape(token)}</mark>")
            spans.append((match.start(), match.end()))
        else:
            pieces.append(html.escape(token))
        cursor = match.end()

    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces), terms, spans


def build_chunk_records(doc_id: str, image_id: int, image_url: str, captions: list[str]) -> tuple[str, list[TextChunk]]:
    unique_captions = list(dict.fromkeys(captions))
    merged_parts: list[str] = []
    chunks: list[TextChunk] = []
    cursor = 0

    for index, caption in enumerate(unique_captions):
        clean_caption = caption.strip()
        if not clean_caption:
            continue
        if merged_parts:
            merged_parts.append(" ")
            cursor += 1
        char_start = cursor
        merged_parts.append(clean_caption)
        cursor += len(clean_caption)
        chunks.append(
            TextChunk(
                chunk_id=f"{doc_id}-chunk-{index}",
                doc_id=doc_id,
                image_id=image_id,
                image_url=image_url,
                chunk_index=index,
                chunk_text=clean_caption,
                char_start=char_start,
                char_end=cursor,
            )
        )

    return "".join(merged_parts), chunks


def build_coco_val_corpus(captions_path: Path, val_images_dir: Path) -> list[CorpusRecord]:
    with captions_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    grouped: dict[int, list[str]] = defaultdict(list)
    for annotation in payload["annotations"]:
        grouped[int(annotation["image_id"])].append(str(annotation["caption"]).strip())

    records: list[CorpusRecord] = []
    for image_id in sorted(grouped):
        file_name = f"{image_id:012d}.jpg"
        image_path = val_images_dir / file_name
        if not image_path.exists():
            continue

        captions = grouped[image_id]
        doc_id = f"coco-val-{image_id}"
        image_url = f"/assets/coco/val2017/{file_name}"
        merged_text, chunks = build_chunk_records(doc_id, image_id, image_url, captions)
        if not chunks:
            continue
        title = chunks[0].chunk_text.rstrip(".")

        records.append(
            CorpusRecord(
                doc_id=doc_id,
                image_id=image_id,
                image_path=str(image_path),
                image_url=image_url,
                image_sha256=compute_image_sha256(image_path),
                image_dhash=compute_image_dhash(image_path),
                title=title,
                merged_text=merged_text,
                captions=[chunk.chunk_text for chunk in chunks],
                keywords=_keywords_from_captions([chunk.chunk_text for chunk in chunks]),
                chunks=chunks,
            )
        )

    return records


def write_corpus_cache(records: list[CorpusRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(record) for record in records], handle, ensure_ascii=True, indent=2)


def read_corpus_cache(input_path: Path) -> list[CorpusRecord]:
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records: list[CorpusRecord] = []
    for item in payload:
        chunks_payload = item.get("chunks", [])
        chunks = [TextChunk(**chunk) for chunk in chunks_payload]

        # Backward compatibility for older caches that stored captions but not chunk metadata.
        if not chunks and item.get("captions"):
            merged_text, rebuilt_chunks = build_chunk_records(
                doc_id=str(item["doc_id"]),
                image_id=int(item["image_id"]),
                image_url=str(item["image_url"]),
                captions=[str(caption) for caption in item.get("captions", [])],
            )
            chunks = rebuilt_chunks
            item = {**item, "merged_text": merged_text}

        item = {**item, "chunks": chunks}
        records.append(CorpusRecord(**item))
    return records
