from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ConfidenceBreakdown(BaseModel):
    retrieval_strength: float = 0.0
    evidence_agreement: float = 0.0
    ranking_margin: float = 0.0
    traceability: float = 0.0
    visual_grounding: float = 0.0
    final_confidence: float = 0.0


class BoundingBox(BaseModel):
    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


class ProofItem(BaseModel):
    id: str
    title: str
    source_kind: Literal[
        "uploaded-image",
        "highlighted-image",
        "detected-region",
        "similar-image",
        "document-page",
        "table-chunk",
        "text-chunk",
    ]
    source_id: str
    image_url: str | None = None
    annotated_image_url: str | None = None
    page_number: int | None = None
    caption: str = ""
    supporting_text: str = ""
    score: float = 0.0
    retrieval_channels: list[str] = Field(default_factory=list)
    channel_scores: dict[str, float] = Field(default_factory=dict)
    matched_terms: list[str] = Field(default_factory=list)
    explanation: list[str] = Field(default_factory=list)
    boxes: list[BoundingBox] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    question: str
    query_kind: Literal["text", "image", "document", "table", "text+image", "document+image"]
    answer: str
    answer_mode: str
    confidence: float
    confidence_breakdown: ConfidenceBreakdown
    answer_explanation: list[str] = Field(default_factory=list)
    proof_summary: str
    highlighted_image_url: str | None = None
    uploaded_file_url: str | None = None
    uploaded_file_name: str | None = None
    proofs: list[ProofItem] = Field(default_factory=list)
    retrieval_trace: dict[str, list[str]] = Field(default_factory=dict)
    corpus_name: str = "Hybrid Multimodal Corpus"


class StatusResponse(BaseModel):
    status: Literal["ok"]
    corpus_name: str
    document_count: int
    document_store_count: int = 0
    vector_store: str
    image_index_source: str
    generator_mode: str
    active_models: dict[str, str]
    capabilities: list[str] = Field(default_factory=list)
    supported_file_types: list[str] = Field(default_factory=list)
    ready: bool = False
    startup_state: Literal["cold", "warming", "ready", "error"] = "cold"
    startup_error: str | None = None
