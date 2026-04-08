from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings, get_settings
from .corpus import matched_terms, tokenize
from .documents import DocumentIndexer
from .generation import GeneratorRouter
from .indexing import CorpusIndexer, RetrievalHit
from .schemas import ConfidenceBreakdown, ProofItem, QueryResponse, StatusResponse
from .vision import VisionAnalysisResult, VisualReasoner


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lower = min(values)
    upper = max(values)
    if abs(upper - lower) < 1e-9:
        return [1.0 for _ in values]
    return [(value - lower) / (upper - lower) for value in values]


@dataclass(slots=True)
class FusedCandidate:
    doc_id: str
    chunk_id: str | None = None
    fusion_score: float = 0.0
    final_score: float = 0.0
    rerank_score: float = 0.0
    raw_channel_scores: dict[str, float] = field(default_factory=dict)
    normalized_channel_scores: dict[str, float] = field(default_factory=dict)
    retrieval_channels: set[str] = field(default_factory=set)
    chunk_signal: float = 0.0


class MultimodalRAGService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.indexer = CorpusIndexer(self.settings)
        self.documents = DocumentIndexer(self.settings)
        self.generator = GeneratorRouter(self.settings)
        self.vision = VisualReasoner(self.settings)

    def warmup(self) -> None:
        try:
            self.indexer.prepare()
        except Exception:
            # Cloud/demo deployments may run without the local COCO corpus.
            # In that case image-question answering can still work through the vision pipeline.
            pass

    def status(self, *, startup_state: str = "ready", startup_error: str | None = None) -> StatusResponse:
        active_models = self.indexer.active_models | {
            "generator": self.generator.active_model_label,
            "vlm": self.settings.vlm_model,
            "detector": self.settings.detector_model,
        }
        image_index_source = self.indexer.image_index_source
        if not self.indexer.is_ready and startup_state != "error":
            image_index_source = "chromadb-pending"
        return StatusResponse(
            status="ok",
            corpus_name="Hybrid Multimodal Corpus",
            document_count=self.indexer.estimated_document_count(),
            document_store_count=self.documents.document_count,
            vector_store=self.indexer.vector_store,
            image_index_source=image_index_source,
            generator_mode=self.generator.generator_mode,
            active_models=active_models,
            capabilities=[
                "zero-shot image reasoning",
                "visual grounding",
                "similar image retrieval",
                "pdf ingestion",
                "table ingestion",
                "confidence scoring",
            ],
            supported_file_types=[".jpg", ".jpeg", ".png", ".webp", ".pdf", ".csv", ".xlsx", ".xls", ".txt"],
            ready=self.indexer.is_ready,
            startup_state="ready" if self.indexer.is_ready else startup_state,
            startup_error=startup_error,
        )

    def ask(self, question: str, image_path: str | None = None) -> QueryResponse:
        normalized_question = question.strip() or "Describe the uploaded content using grounded proof."
        file_kind = self._file_kind(image_path)

        if file_kind in {"pdf", "table", "text-document"}:
            return self._ask_document(normalized_question, image_path)
        if file_kind == "image":
            return self._ask_image(normalized_question, image_path)
        return self._ask_text(normalized_question)

    def _ask_text(self, question: str) -> QueryResponse:
        try:
            self.indexer.ensure_corpus_ready()
        except Exception:
            return QueryResponse(
                question=question,
                query_kind="text",
                answer="Text-only retrieval is not available in this deployment because the indexed corpus is not present. Upload an image and ask a question to use the live demo.",
                answer_mode="fallback-no-corpus",
                confidence=0.0,
                confidence_breakdown=ConfidenceBreakdown(),
                answer_explanation=[
                    "The deployment is running without the local indexed corpus.",
                    "Image-based reasoning is still available through the vision pipeline.",
                ],
                proof_summary="No indexed corpus was available for text-only retrieval.",
                proofs=[],
                retrieval_trace={},
                corpus_name="Cloud Demo",
            )
        proofs, trace = self._corpus_proofs(question=question, image_path=None, allow_clip=True)
        generation = self.generator.generate(question, proofs)
        confidence = self._compute_confidence(proofs, visual_grounding=0.0)
        return QueryResponse(
            question=question,
            query_kind="text",
            answer=generation.answer,
            answer_mode=generation.mode,
            confidence=confidence.final_confidence,
            confidence_breakdown=confidence,
            answer_explanation=self._answer_explanation(proofs, confidence),
            proof_summary=self._proof_summary(proofs),
            proofs=proofs,
            retrieval_trace=trace,
            corpus_name="COCO 2017 Validation",
        )

    def _ask_image(self, question: str, image_path: str) -> QueryResponse:
        visual = self.vision.analyze(image_path, question)
        try:
            self.indexer.ensure_corpus_ready()
            similar_proofs, trace = self._corpus_proofs(question=question, image_path=image_path, allow_clip=True)
        except Exception:
            similar_proofs, trace = [], {}
        proofs = self._dedupe_proofs(visual.proofs + similar_proofs)[: max(self.settings.visual_proof_count + 2, self.settings.top_k)]
        answer = visual.answer
        answer_mode = visual.answer_mode
        retrieval_override_note = None
        lower_question = question.lower()
        generic_image_query = self._is_generic_image_query(question)
        prefers_direct_visual_answer = any(
            [
                generic_image_query,
                "describe" in lower_question,
                "how many" in lower_question,
                "what objects" in lower_question,
                "what is present" in lower_question,
                "what are present" in lower_question,
                "list the objects" in lower_question,
            ]
        )
        if similar_proofs:
            lead_similar = similar_proofs[0]
            has_strong_image_match = bool(
                {"exact_image_match", "near_image_match", "clip_image"}.intersection(lead_similar.retrieval_channels)
            )
            if has_strong_image_match and lead_similar.score >= 0.86 and not prefers_direct_visual_answer:
                grounded = self.generator.generate(question, similar_proofs[:2])
                answer = grounded.answer
                answer_mode = f"{visual.answer_mode}+retrieval"
                retrieval_override_note = (
                    f"High-confidence retrieved image evidence from {lead_similar.id} was used to refine the visual answer."
                )
        confidence = self._compute_confidence(proofs, visual_grounding=visual.visual_grounding_score)
        answer_explanation = list(visual.explanation)
        if retrieval_override_note is not None:
            answer_explanation.append(retrieval_override_note)
        answer_explanation.extend(self._answer_explanation(proofs, confidence))
        uploaded_url = self._file_url(image_path)
        return QueryResponse(
            question=question,
            query_kind="text+image" if question.strip() else "image",
            answer=answer,
            answer_mode=answer_mode,
            confidence=confidence.final_confidence,
            confidence_breakdown=confidence,
            answer_explanation=answer_explanation,
            proof_summary=self._proof_summary(proofs, visual=visual),
            highlighted_image_url=visual.highlighted_image_url,
            uploaded_file_url=uploaded_url,
            uploaded_file_name=Path(image_path).name,
            proofs=proofs,
            retrieval_trace=self._visual_trace(proofs, trace),
            corpus_name="COCO 2017 Validation + Uploaded Visual Proof",
        )

    def _ask_document(self, question: str, file_path: str) -> QueryResponse:
        proofs = self.documents.search(file_path, question, top_k=max(4, self.settings.top_k))
        generation = self.generator.generate(question, proofs)
        confidence = self._compute_confidence(proofs, visual_grounding=0.0)
        query_kind = "table" if self._file_kind(file_path) == "table" else "document"
        return QueryResponse(
            question=question,
            query_kind=query_kind,
            answer=generation.answer,
            answer_mode=generation.mode,
            confidence=confidence.final_confidence,
            confidence_breakdown=confidence,
            answer_explanation=self._answer_explanation(proofs, confidence),
            proof_summary=self._proof_summary(proofs),
            uploaded_file_url=self._file_url(file_path),
            uploaded_file_name=Path(file_path).name,
            proofs=proofs,
            retrieval_trace={proof.id: proof.retrieval_channels for proof in proofs},
            corpus_name="Uploaded Document Store",
        )

    def _corpus_proofs(self, question: str, image_path: str | None, allow_clip: bool) -> tuple[list[ProofItem], dict[str, list[str]]]:
        has_image = image_path is not None
        has_user_question = bool(question.strip())
        generic_image_query = self._is_generic_image_query(question)
        use_text_channels = not has_image or (has_user_question and not generic_image_query)
        full_index_ready = self.indexer.is_ready
        top_k = max(self.settings.top_k, 4)
        fusion_k = self.settings.query_fusion_k

        channel_runs: dict[str, list[RetrievalHit]] = {}
        if use_text_channels:
            channel_runs["bm25"] = self.indexer.search_bm25(question, fusion_k)
            if full_index_ready:
                channel_runs["dense_text"] = self.indexer.search_dense_text(question, fusion_k)
                if allow_clip:
                    channel_runs["clip_text"] = self.indexer.search_clip_text(question, fusion_k)

        if image_path:
            exact_match = self.indexer.find_exact_image_match(image_path)
            if exact_match is not None:
                channel_runs["exact_image_match"] = [exact_match]
            near_matches = self.indexer.search_near_image_matches(image_path, top_k=min(max(top_k, 3), fusion_k))
            if near_matches:
                channel_runs["near_image_match"] = near_matches
            if full_index_ready and allow_clip:
                channel_runs["clip_image"] = self.indexer.search_clip_image(image_path, fusion_k)

        fused: dict[str, FusedCandidate] = defaultdict(lambda: FusedCandidate(doc_id=""))
        weights = self._channel_weights(has_image=has_image, generic_image_query=generic_image_query)
        for channel, hits in channel_runs.items():
            normalized_scores = _minmax([hit.score for hit in hits])
            for rank, (hit, normalized_score) in enumerate(zip(hits, normalized_scores), start=1):
                rank_score = 1.0 / (rank + 1.0)
                contribution = weights.get(channel, 0.5) * ((0.72 * normalized_score) + (0.28 * rank_score))
                candidate = fused[hit.doc_id]
                candidate.doc_id = hit.doc_id
                candidate.fusion_score += contribution
                candidate.raw_channel_scores[channel] = hit.score
                candidate.normalized_channel_scores[channel] = normalized_score
                candidate.retrieval_channels.add(channel)
                if hit.chunk_id and normalized_score >= candidate.chunk_signal:
                    candidate.chunk_id = hit.chunk_id
                    candidate.chunk_signal = normalized_score

        sorted_candidates = sorted(fused.values(), key=lambda item: item.fusion_score, reverse=True)[: max(top_k * 3, 10)]
        normalized_fusion = _minmax([candidate.fusion_score for candidate in sorted_candidates])
        for candidate, fusion_score in zip(sorted_candidates, normalized_fusion):
            candidate.fusion_score = fusion_score
            if candidate.chunk_id is None or {
                "exact_image_match",
                "near_image_match",
                "clip_image",
            }.intersection(candidate.retrieval_channels):
                candidate.chunk_id = self.indexer.resolve_support_chunk(candidate.doc_id, question).chunk_id

        rerank_weight = self._rerank_weight(
            has_image=has_image,
            has_user_question=has_user_question,
            generic_image_query=generic_image_query,
        )
        rerank_scores = {}
        if self.settings.use_reranker and full_index_ready and rerank_weight > 0.0 and sorted_candidates:
            chunks = [self.indexer.get_chunk(candidate.chunk_id) for candidate in sorted_candidates if candidate.chunk_id]
            rerank_scores = self.indexer.rerank(question, chunks)

        ranked_pairs = []
        for candidate in sorted_candidates:
            rerank = rerank_scores.get(candidate.chunk_id or "", 0.0)
            candidate.rerank_score = rerank
            image_signal = max(
                candidate.normalized_channel_scores.get("exact_image_match", 0.0),
                candidate.normalized_channel_scores.get("near_image_match", 0.0),
                candidate.normalized_channel_scores.get("clip_image", 0.0),
            )
            final_score = ((1.0 - rerank_weight) * candidate.fusion_score) + (rerank_weight * rerank)
            if has_image:
                final_score = (0.82 * final_score) + (0.18 * image_signal)
            if "exact_image_match" in candidate.retrieval_channels:
                final_score = max(final_score, 0.995)
            elif "near_image_match" in candidate.retrieval_channels:
                final_score = max(final_score, 0.9 + (0.05 * image_signal))
            candidate.final_score = min(0.999, final_score)
            ranked_pairs.append((candidate, candidate.final_score))

        ranked_pairs.sort(key=lambda pair: pair[1], reverse=True)
        top_pairs = ranked_pairs[:top_k]
        proofs: list[ProofItem] = []
        trace: dict[str, list[str]] = {}
        for index, (candidate, _) in enumerate(top_pairs, start=1):
            record = self.indexer.corpus_by_id[candidate.doc_id]
            chunk = self.indexer.get_chunk(candidate.chunk_id or record.chunks[0].chunk_id)
            proof_id = f"S{index}"
            channels = sorted(candidate.retrieval_channels)
            trace[proof_id] = channels
            proofs.append(
                ProofItem(
                    id=proof_id,
                    title=record.title,
                    source_kind="similar-image",
                    source_id=record.doc_id,
                    image_url=record.image_url,
                    caption=record.captions[0],
                    supporting_text=chunk.chunk_text,
                    score=round(float(candidate.final_score), 4),
                    retrieval_channels=channels,
                    channel_scores={key: round(float(value), 4) for key, value in candidate.raw_channel_scores.items()},
                    matched_terms=matched_terms(question, chunk.chunk_text),
                    explanation=self._corpus_proof_explanation(candidate, chunk),
                    metadata={"image_id": record.image_id, "keywords": record.keywords},
                )
            )
        return proofs, trace

    def _corpus_proof_explanation(self, candidate: FusedCandidate, chunk) -> list[str]:
        lines = [
            f"Source chunk: {chunk.chunk_id}.",
            f"Retrieval channels: {', '.join(sorted(candidate.retrieval_channels))}.",
        ]
        if "exact_image_match" in candidate.retrieval_channels:
            lines.append("The uploaded image matches this indexed image exactly.")
        elif "near_image_match" in candidate.retrieval_channels:
            lines.append("The uploaded image is visually near this indexed example.")
        else:
            lines.append("This proof is one of the closest retrieved matches from the indexed corpus.")
        return lines

    def _compute_confidence(self, proofs: list[ProofItem], visual_grounding: float) -> ConfidenceBreakdown:
        if not proofs:
            return ConfidenceBreakdown()

        top_scores = [proof.score for proof in proofs[:4]]
        retrieval_strength = max(top_scores[0], sum(top_scores) / len(top_scores))
        evidence_agreement = sum(min(1.0, len(proof.retrieval_channels) / 3.0) for proof in proofs[:4]) / len(top_scores)
        if len(top_scores) > 1:
            margin = max(0.0, top_scores[0] - top_scores[1])
        else:
            margin = max(0.0, top_scores[0] * 0.25)
        ranking_margin = min(1.0, margin / 0.35)
        traceability = sum(
            1.0 if proof.image_url or proof.supporting_text or proof.boxes else 0.0 for proof in proofs[:4]
        ) / len(top_scores)
        final_confidence = (
            (0.32 * retrieval_strength)
            + (0.18 * evidence_agreement)
            + (0.15 * ranking_margin)
            + (0.15 * traceability)
            + (0.20 * visual_grounding)
        )
        return ConfidenceBreakdown(
            retrieval_strength=round(min(0.999, retrieval_strength), 3),
            evidence_agreement=round(min(0.999, evidence_agreement), 3),
            ranking_margin=round(min(0.999, ranking_margin), 3),
            traceability=round(min(0.999, traceability), 3),
            visual_grounding=round(min(0.999, visual_grounding), 3),
            final_confidence=round(min(0.999, final_confidence), 3),
        )

    def _answer_explanation(self, proofs: list[ProofItem], confidence: ConfidenceBreakdown) -> list[str]:
        if not proofs:
            return ["No grounded proof was available for this answer."]
        lead = proofs[0]
        lines = [
            f"The answer is grounded primarily in {lead.id} from source `{lead.source_id}`.",
            f"{lead.id} was selected through {', '.join(lead.retrieval_channels) or 'proof synthesis'} with score {lead.score:.3f}.",
            (
                "Confidence is computed programmatically from retrieval strength, evidence agreement, ranking margin, "
                f"traceability, and visual grounding: {confidence.final_confidence:.3f}."
            ),
        ]
        if len(proofs) > 1:
            lines.append("Additional supporting proofs: " + ", ".join(proof.id for proof in proofs[1:4]) + ".")
        return lines

    def _proof_summary(self, proofs: list[ProofItem], visual: VisionAnalysisResult | None = None) -> str:
        if visual is not None:
            labels = ", ".join(box.label for box in visual.detected_boxes[:4]) or "no strong detections"
            return f"Primary proof comes from grounded visual regions in the uploaded image. Key highlighted objects: {labels}."
        if not proofs:
            return "No proof could be retrieved."
        lead = proofs[0]
        if lead.source_kind == "document-page":
            return f"Top proof comes from page {lead.page_number} of the uploaded document."
        if lead.source_kind == "table-chunk":
            return "Top proof comes from the uploaded table content."
        return f"Top proof is {lead.id}, retrieved through {', '.join(lead.retrieval_channels) or 'proof synthesis'}."

    def _visual_trace(self, proofs: list[ProofItem], corpus_trace: dict[str, list[str]]) -> dict[str, list[str]]:
        trace = dict(corpus_trace)
        for proof in proofs:
            trace.setdefault(proof.id, proof.retrieval_channels)
        return trace

    def _dedupe_proofs(self, proofs: list[ProofItem]) -> list[ProofItem]:
        unique: list[ProofItem] = []
        seen: set[tuple[str, str]] = set()
        for proof in proofs:
            key = (proof.source_kind, proof.source_id)
            if key in seen:
                continue
            seen.add(key)
            unique.append(proof)
        return unique

    def _file_url(self, file_path: str | None) -> str | None:
        if not file_path:
            return None
        path = Path(file_path).resolve()
        try:
            relative = path.relative_to(self.settings.upload_dir.resolve())
            return f"/uploads/{relative.as_posix()}"
        except ValueError:
            return None

    def _file_kind(self, file_path: str | None) -> str:
        if not file_path:
            return "text"
        suffix = Path(file_path).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
            return "image"
        if suffix == ".pdf":
            return "pdf"
        if suffix in {".csv", ".xlsx", ".xls"}:
            return "table"
        return "text-document"

    def _channel_weights(self, *, has_image: bool, generic_image_query: bool) -> dict[str, float]:
        weights = {
            "dense_text": 1.15,
            "bm25": 0.8,
            "clip_text": 0.95,
            "clip_image": 1.35,
            "exact_image_match": 4.2,
            "near_image_match": 2.8,
        }
        if has_image and generic_image_query:
            return {
                "dense_text": 0.15,
                "bm25": 0.1,
                "clip_text": 0.25,
                "clip_image": 2.25,
                "exact_image_match": 4.8,
                "near_image_match": 3.1,
            }
        if has_image:
            weights["dense_text"] = 0.7
            weights["bm25"] = 0.45
            weights["clip_text"] = 0.8
            weights["clip_image"] = 1.75
        return weights

    def _rerank_weight(self, *, has_image: bool, has_user_question: bool, generic_image_query: bool) -> float:
        if not self.settings.use_reranker:
            return 0.0
        if not has_image:
            return 0.55
        if not has_user_question:
            return 0.0
        if generic_image_query:
            return 0.1
        return 0.28

    def _is_generic_image_query(self, question: str) -> bool:
        normalized = " ".join(tokenize(question))
        if not normalized:
            return True
        generic_phrases = {
            "describe this image",
            "describe the image",
            "describe this picture",
            "describe the picture",
            "describe this photo",
            "describe the photo",
            "identify this image",
            "identify this picture",
            "what is this",
            "what is in this image",
            "what is in the image",
            "what is shown in this image",
            "what is shown in the image",
            "what do you see",
        }
        if normalized in generic_phrases:
            return True
        generic_prefixes = ("describe ", "identify ", "what is in ", "what is shown in ", "what do you see")
        if any(normalized.startswith(prefix) for prefix in generic_prefixes):
            return len(tokenize(normalized)) <= 7
        tokens = tokenize(normalized)
        return len(tokens) <= 5 and bool({"image", "picture", "photo"}.intersection(tokens))
