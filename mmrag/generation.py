from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .schemas import ProofItem


@dataclass(slots=True)
class GenerationResult:
    answer: str
    mode: str


class ExtractiveGroundedGenerator:
    def generate(self, question: str, proofs: list[ProofItem]) -> GenerationResult:
        if not proofs:
            return GenerationResult(
                answer="I could not find grounded proof for the request.",
                mode="extractive-fallback",
            )

        lead = proofs[0]
        support = proofs[1] if len(proofs) > 1 else None
        question_lower = question.strip().lower()
        support_text = lead.supporting_text or lead.caption or lead.title

        if lead.source_kind in {"detected-region", "highlighted-image", "uploaded-image"}:
            answer = f"The uploaded visual evidence suggests: {support_text} [{lead.id}]."
        elif question_lower.startswith(("is ", "are ", "does ", "do ", "can ", "could ", "was ", "were ")):
            answer = f"The retrieved proof suggests yes: {support_text} [{lead.id}]."
        elif question_lower.startswith(("who ", "what ", "where ", "why ", "how ")):
            answer = f"The strongest grounded match is: {support_text} [{lead.id}]."
        else:
            answer = f"The best grounded answer is supported by: {support_text} [{lead.id}]."

        if support is not None and support.score >= 0.2:
            answer += f" Additional support comes from {support.supporting_text or support.caption or support.title} [{support.id}]."

        return GenerationResult(answer=answer, mode="extractive-fallback")


class OpenAIGroundedGenerator:
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def generate(self, question: str, proofs: list[ProofItem]) -> GenerationResult:
        lines = []
        for proof in proofs:
            lines.append(
                "\n".join(
                    [
                        f"[{proof.id}] title: {proof.title}",
                        f"[{proof.id}] source_kind: {proof.source_kind}",
                        f"[{proof.id}] supporting_text: {proof.supporting_text or proof.caption}",
                        f"[{proof.id}] explanation: {' '.join(proof.explanation)}",
                    ]
                )
            )

        prompt = (
            "Answer only from the grounded proof items below.\n"
            "Do not invent facts.\n"
            "Cite proof ids inline like [P1] or [D1].\n\n"
            f"Question:\n{question}\n\n"
            f"Proofs:\n{'\n\n'.join(lines)}"
        )

        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        )
        answer = response.output_text.strip()
        if proofs and "[" not in answer:
            answer = f"{answer} [{proofs[0].id}]"
        return GenerationResult(answer=answer, mode=f"openai:{self.model}")


class GeneratorRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.extractive = ExtractiveGroundedGenerator()
        self.openai = None
        self._openai_failed = False

    def _wants_openai(self) -> bool:
        return (
            self.settings.llm_provider == "openai"
            and self.settings.use_openai_if_available
            and bool(self.settings.openai_api_key)
        )

    def _get_openai(self) -> OpenAIGroundedGenerator | None:
        if not self._wants_openai() or self._openai_failed:
            return None
        if self.openai is None:
            try:
                self.openai = OpenAIGroundedGenerator(self.settings)
            except Exception:
                self._openai_failed = True
                self.openai = None
        return self.openai

    @property
    def generator_mode(self) -> str:
        if self._wants_openai() and not self._openai_failed:
            return "openai"
        return "extractive-fallback"

    @property
    def active_model_label(self) -> str:
        if self._wants_openai() and not self._openai_failed:
            return self.settings.openai_model
        return "extractive-fallback"

    def generate(self, question: str, proofs: list[ProofItem]) -> GenerationResult:
        openai_generator = self._get_openai()
        if openai_generator is not None:
            try:
                return openai_generator.generate(question, proofs)
            except Exception:
                pass
        return self.extractive.generate(question, proofs)
