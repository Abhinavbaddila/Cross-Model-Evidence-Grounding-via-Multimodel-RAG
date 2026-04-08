from __future__ import annotations

import base64
import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .config import Settings
from .corpus import salient_terms
from .schemas import BoundingBox, ProofItem

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None


COMMON_OBJECT_LABELS = [
    "person",
    "people",
    "man",
    "woman",
    "child",
    "boy",
    "girl",
    "player",
    "sports ball",
    "football",
    "soccer ball",
    "basketball",
    "tennis racket",
    "baseball bat",
    "baseball glove",
    "frisbee",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "train",
    "airplane",
    "boat",
    "dog",
    "cat",
    "bird",
    "horse",
    "cow",
    "sheep",
    "bear",
    "zebra",
    "giraffe",
    "chair",
    "couch",
    "dining table",
    "table",
    "bench",
    "bed",
    "tv",
    "laptop",
    "cell phone",
    "book",
    "backpack",
    "handbag",
    "suitcase",
    "bottle",
    "cup",
    "bowl",
    "banana",
    "apple",
    "orange",
    "pizza",
    "cake",
    "sandwich",
    "hot dog",
    "donut",
    "clock",
    "toilet",
    "sink",
    "refrigerator",
    "oven",
    "microwave",
    "traffic light",
    "stop sign",
    "fire hydrant",
    "parking meter",
    "chart",
    "document",
    "page",
]


LABEL_ALIASES = {
    "people": "person",
    "man": "person",
    "woman": "person",
    "child": "person",
    "boy": "person",
    "girl": "person",
    "player": "person",
    "soccer ball": "sports ball",
    "football": "sports ball",
    "tvmonitor": "tv",
    "cellphone": "cell phone",
    "diningtable": "dining table",
    "pottedplant": "potted plant",
}


ACTION_PATTERNS = [
    ("playing tennis", {"person", "tennis racket", "sports ball"}),
    ("playing baseball", {"person", "baseball bat", "baseball glove", "sports ball"}),
    ("throwing or catching a frisbee", {"person", "frisbee"}),
    ("riding a bicycle", {"person", "bicycle"}),
    ("riding a motorcycle", {"person", "motorcycle"}),
    ("surfing", {"person", "surfboard"}),
    ("skateboarding", {"person", "skateboard"}),
    ("snowboarding", {"person", "snowboard"}),
    ("skiing", {"person", "skis"}),
    ("using a laptop", {"person", "laptop"}),
    ("using a phone", {"person", "cell phone"}),
    ("reading", {"person", "book"}),
    ("eating or dining", {"person", "dining table", "cup", "bottle", "bowl"}),
    ("playing football", {"person", "sports ball"}),
    ("sitting", {"person", "chair", "bench", "couch"}),
]


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "item"


def _pluralize(label: str, count: int) -> str:
    if count == 1:
        return label
    if label.endswith("y") and not label.endswith("ay"):
        return label[:-1] + "ies"
    if label.endswith("s"):
        return label
    return label + "s"


def _normalize_label(label: str) -> str:
    normalized = re.sub(r"\s+", " ", label.strip().lower())
    return LABEL_ALIASES.get(normalized, normalized)


def _data_url_for_image(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    payload = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    return f"data:{media_type};base64,{payload}"


def _counts_from_detections(detections: list[BoundingBox]) -> Counter[str]:
    return Counter(_normalize_label(box.label) for box in detections)


def _top_labels(detections: list[BoundingBox], limit: int = 6) -> list[str]:
    labels: list[str] = []
    for box in detections:
        label = _normalize_label(box.label)
        if label not in labels:
            labels.append(label)
        if len(labels) >= limit:
            break
    return labels


def _render_caption_from_detections(detections: list[BoundingBox]) -> str:
    counts = _counts_from_detections(detections)
    if not counts:
        return "An uploaded image."

    phrases = [f"{count} {_pluralize(label, count)}" for label, count in counts.most_common(5)]
    if len(phrases) == 1:
        return f"The image shows {phrases[0]}."
    return f"The image shows {', '.join(phrases[:-1])}, and {phrases[-1]}."


def guess_object_labels(question: str, caption: str, limit: int = 12) -> list[str]:
    tokens = salient_terms(question) | salient_terms(caption)
    labels: list[str] = []
    for label in COMMON_OBJECT_LABELS:
        label_tokens = set(label.split())
        if label in tokens or tokens.intersection(label_tokens):
            labels.append(label)
    if "people" in labels and "person" not in labels:
        labels.insert(0, "person")
    if not labels:
        labels = ["person", "car", "dog", "cat", "table", "chair", "bicycle", "cell phone"]
    return list(dict.fromkeys(labels))[:limit]


def draw_boxes(image_path: str, boxes: list[BoundingBox], output_path: Path) -> None:
    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        for index, box in enumerate(boxes, start=1):
            outline = "#FF6B35" if index == 1 else "#0F766E"
            draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=outline, width=4)
            label = f"{box.label} {box.score:.2f}"
            anchor_y = max(0, box.y1 - 22)
            draw.rectangle((box.x1, anchor_y, box.x1 + 180, anchor_y + 20), fill=outline)
            draw.text((box.x1 + 6, anchor_y + 2), label, fill="white")
        image.save(output_path)


def save_crop(image_path: str, box: BoundingBox, output_path: Path) -> None:
    with Image.open(image_path).convert("RGB") as image:
        x1 = max(0, int(math.floor(box.x1)))
        y1 = max(0, int(math.floor(box.y1)))
        x2 = min(image.width, int(math.ceil(box.x2)))
        y2 = min(image.height, int(math.ceil(box.y2)))
        crop = image.crop((x1, y1, x2, y2))
        crop.save(output_path)


def best_count_target(question: str, detections: list[BoundingBox]) -> str | None:
    normalized = question.lower()
    if "how many" not in normalized:
        return None
    for label in COMMON_OBJECT_LABELS:
        if label in normalized:
            canonical = _normalize_label(label)
            if canonical in {"people", "man", "woman", "boy", "girl", "player"}:
                return "person"
            return canonical
    if detections:
        return _normalize_label(detections[0].label)
    return None


def _looks_like_action_question(question: str) -> bool:
    normalized = question.lower()
    return any(
        phrase in normalized
        for phrase in [
            "what is the person doing",
            "what are the people doing",
            "what are they doing",
            "what is happening",
            "what activity",
            "doing in this image",
        ]
    )


def _looks_like_objects_question(question: str) -> bool:
    normalized = question.lower()
    return any(
        phrase in normalized
        for phrase in ["what objects", "what is present", "what are present", "what can you see", "list the objects"]
    )


def _infer_activity(detections: list[BoundingBox]) -> tuple[str | None, set[str]]:
    counts = _counts_from_detections(detections)
    if counts.get("person", 0) == 0:
        return None, set()

    for activity, labels in ACTION_PATTERNS:
        if any(counts.get(label, 0) > 0 for label in labels if label != "person"):
            return activity, labels
    return "standing", {"person"}


def _focus_labels_for_question(question: str, detections: list[BoundingBox]) -> set[str]:
    count_target = best_count_target(question, detections)
    if count_target is not None:
        return {count_target}

    if _looks_like_action_question(question):
        _, labels = _infer_activity(detections)
        return labels or {"person"}

    labels = {
        _normalize_label(label)
        for label in COMMON_OBJECT_LABELS
        if label in question.lower()
    }
    if labels:
        return labels

    if _looks_like_objects_question(question):
        return set(_top_labels(detections))

    return set(_top_labels(detections, limit=4))


def _box_question_score(box: BoundingBox, question: str, focus_labels: set[str]) -> float:
    label = _normalize_label(box.label)
    score = float(box.score)
    if label in focus_labels:
        score += 0.65
    if label in question.lower():
        score += 0.25
    if label == "person" and _looks_like_action_question(question):
        score += 0.3
    return score


def _select_proof_boxes(question: str, detections: list[BoundingBox], limit: int) -> list[BoundingBox]:
    if not detections:
        return []

    focus_labels = _focus_labels_for_question(question, detections)
    count_target = best_count_target(question, detections)
    if count_target is not None:
        matches = [box for box in detections if _normalize_label(box.label) == count_target]
        return matches[:limit] or detections[:limit]

    if _looks_like_objects_question(question):
        selected: list[BoundingBox] = []
        seen_labels: set[str] = set()
        for box in sorted(detections, key=lambda item: _box_question_score(item, question, focus_labels), reverse=True):
            label = _normalize_label(box.label)
            if label in seen_labels:
                continue
            selected.append(box)
            seen_labels.add(label)
            if len(selected) >= limit:
                break
        return selected or detections[:limit]

    ranked = sorted(detections, key=lambda item: _box_question_score(item, question, focus_labels), reverse=True)
    return ranked[:limit]


def _answer_from_rules(question: str, caption: str, detections: list[BoundingBox]) -> tuple[str, str]:
    count_target = best_count_target(question, detections)
    if count_target is not None:
        count = sum(1 for box in detections if _normalize_label(box.label) == count_target)
        if count:
            return f"There are {count} {_pluralize(count_target, count)} visible in the image.", "rule-count"

    if _looks_like_objects_question(question) and detections:
        labels = _top_labels(detections)
        return f"The visible objects include {', '.join(labels)}.", "rule-objects"

    if _looks_like_action_question(question):
        activity, _ = _infer_activity(detections)
        if activity is not None:
            people = _counts_from_detections(detections).get("person", 0)
            if people > 1:
                return f"The people appear to be {activity}.", "rule-activity"
            return f"The person appears to be {activity}.", "rule-activity"
        if detections:
            return f"The image shows {caption.lower()}", "caption+detector"

    if question.lower().startswith(("is there", "are there", "does the image show")):
        labels = _focus_labels_for_question(question, detections)
        if labels:
            found = any(_normalize_label(box.label) in labels for box in detections)
            prefix = "Yes" if found else "No"
            if found:
                return f"{prefix}, the image contains {', '.join(sorted(labels))}.", "rule-boolean"
            return f"{prefix}, the requested object is not clearly detected.", "rule-boolean"

    if detections:
        return caption, "detector-caption"
    return "I could not detect confident objects, but I can still use the uploaded image as context.", "fallback-caption"


def _prune_detections(detections: list[BoundingBox], threshold: float) -> list[BoundingBox]:
    if not detections:
        return []
    dynamic_threshold = max(threshold, float(detections[0].score) * 0.45)
    return [box for box in detections if box.score >= dynamic_threshold][:12]


@dataclass(slots=True)
class VisionAnalysisResult:
    answer: str
    answer_mode: str
    caption: str
    highlighted_image_url: str
    proofs: list[ProofItem]
    visual_grounding_score: float
    explanation: list[str]
    detected_boxes: list[BoundingBox]


class OpenAIVisionBackend:
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def answer_question(self, image_path: str, question: str, caption: str, detections: list[BoundingBox]) -> str:
        detection_summary = ", ".join(f"{box.label} ({box.score:.2f})" for box in detections[:6]) or "no strong detections"
        prompt = (
            "You are answering a visual question about an uploaded image.\n"
            "Use the image first. Use detections only as supporting hints.\n"
            "Be concise, factual, and grounded.\n\n"
            f"Question: {question or 'Describe the image.'}\n"
            f"Detection summary: {detection_summary}\n"
            f"Caption hint: {caption}"
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": _data_url_for_image(image_path)},
                    ],
                }
            ],
        )
        return response.output_text.strip()


class LlavaVisionBackend:
    def __init__(self, settings: Settings) -> None:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = settings.vlm_model
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def answer_question(self, image_path: str, question: str) -> str:
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        with Image.open(image_path).convert("RGB") as image:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=64)
        text = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        if "ASSISTANT:" in text:
            return text.split("ASSISTANT:", 1)[-1].strip()
        return text


class BlipVisionBackend:
    def __init__(self, settings: Settings) -> None:
        from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering, BlipProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.caption_processor = BlipProcessor.from_pretrained(settings.caption_model)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(settings.caption_model).to(self.device)
        self.vqa_processor = BlipProcessor.from_pretrained(settings.vqa_model)
        self.vqa_model = BlipForQuestionAnswering.from_pretrained(settings.vqa_model).to(self.device)
        self.caption_model.eval()
        self.vqa_model.eval()

    @torch.no_grad()
    def caption(self, image_path: str) -> str:
        with Image.open(image_path).convert("RGB") as image:
            inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
            output = self.caption_model.generate(**inputs, max_new_tokens=48)
        return self.caption_processor.decode(output[0], skip_special_tokens=True).strip()

    @torch.no_grad()
    def answer_question(self, image_path: str, question: str) -> str:
        with Image.open(image_path).convert("RGB") as image:
            inputs = self.vqa_processor(images=image, text=question, return_tensors="pt").to(self.device)
            output = self.vqa_model.generate(**inputs, max_new_tokens=32)
        return self.vqa_processor.decode(output[0], skip_special_tokens=True).strip()


class YoloDetector:
    def __init__(self, model_name: str) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed")
        self.model = YOLO(model_name)

    def detect(self, image_path: str, labels: list[str], threshold: float) -> list[BoundingBox]:
        results = self.model.predict(
            source=image_path,
            conf=threshold,
            imgsz=1280,
            verbose=False,
            max_det=24,
        )
        if not results:
            return []

        result = results[0]
        names = result.names
        raw_boxes = getattr(result, "boxes", None)
        if raw_boxes is None:
            return []

        xyxy = raw_boxes.xyxy.cpu().tolist()
        scores = raw_boxes.conf.cpu().tolist()
        classes = raw_boxes.cls.cpu().tolist()
        boxes: list[BoundingBox] = []
        for coords, score, class_id in zip(xyxy, scores, classes):
            label = _normalize_label(str(names[int(class_id)]))
            boxes.append(
                BoundingBox(
                    label=label,
                    score=float(score),
                    x1=float(coords[0]),
                    y1=float(coords[1]),
                    x2=float(coords[2]),
                    y2=float(coords[3]),
                )
            )
        return sorted(boxes, key=lambda item: item.score, reverse=True)


class GroundingDinoDetector:
    def __init__(self, model_name: str) -> None:
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = GroundingDinoProcessor.from_pretrained(model_name)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, image_path: str, labels: list[str], threshold: float) -> list[BoundingBox]:
        labels = labels or ["object"]
        with Image.open(image_path).convert("RGB") as image:
            inputs = self.processor(images=image, text=labels, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=threshold,
                text_threshold=threshold,
                target_sizes=[image.size[::-1]],
                text_labels=[labels],
            )[0]
        boxes: list[BoundingBox] = []
        for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
            boxes.append(
                BoundingBox(
                    label=_normalize_label(str(label)),
                    score=float(score),
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                )
            )
        return sorted(boxes, key=lambda item: item.score, reverse=True)


class Owlv2Detector:
    def __init__(self, model_name: str) -> None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, image_path: str, labels: list[str], threshold: float) -> list[BoundingBox]:
        labels = labels or ["object"]
        with Image.open(image_path).convert("RGB") as image:
            inputs = self.processor(text=[labels], images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                threshold=threshold,
                target_sizes=[image.size[::-1]],
                text_labels=[labels],
            )[0]
        boxes: list[BoundingBox] = []
        for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
            boxes.append(
                BoundingBox(
                    label=_normalize_label(str(label)),
                    score=float(score),
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                )
            )
        return sorted(boxes, key=lambda item: item.score, reverse=True)


class VisualReasoner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._openai_backend: OpenAIVisionBackend | None = None
        self._openai_failed = False
        self._llava_backend: LlavaVisionBackend | None = None
        self._llava_failed = False
        self._blip_backend: BlipVisionBackend | None = None
        self._blip_failed = False
        self._detector = None
        self._detector_failed = False

    def _get_openai_backend(self) -> OpenAIVisionBackend | None:
        if not self.settings.openai_api_key or self._openai_failed:
            return None
        if self._openai_backend is None:
            try:
                self._openai_backend = OpenAIVisionBackend(self.settings)
            except Exception:
                self._openai_failed = True
                self._openai_backend = None
        return self._openai_backend

    def _get_llava_backend(self) -> LlavaVisionBackend | None:
        if self._llava_failed:
            return None
        if self._llava_backend is None:
            try:
                self._llava_backend = LlavaVisionBackend(self.settings)
            except Exception:
                self._llava_failed = True
                self._llava_backend = None
        return self._llava_backend

    def _get_blip_backend(self) -> BlipVisionBackend | None:
        if self._blip_failed:
            return None
        if self._blip_backend is None:
            try:
                self._blip_backend = BlipVisionBackend(self.settings)
            except Exception:
                self._blip_failed = True
                self._blip_backend = None
        return self._blip_backend

    def _get_detector(self):
        if self._detector_failed:
            return None
        if self._detector is None:
            try:
                provider = self.settings.detector_provider
                if provider in {"auto", "yolo"}:
                    self._detector = YoloDetector(self.settings.detector_model)
                elif provider in {"grounding-dino", "grounding_dino"}:
                    self._detector = GroundingDinoDetector(self.settings.detector_model)
                else:
                    self._detector = Owlv2Detector(self.settings.detector_fallback_model)
            except Exception:
                try:
                    self._detector = Owlv2Detector(self.settings.detector_fallback_model)
                except Exception:
                    self._detector_failed = True
                    self._detector = None
        return self._detector

    def _answer_with_optional_vlm(
        self,
        image_path: str,
        question: str,
        caption: str,
        detections: list[BoundingBox],
    ) -> tuple[str | None, str | None]:
        provider = self.settings.vlm_provider
        normalized_question = question.strip() or "Describe the image."

        if provider == "openai":
            backend = self._get_openai_backend()
            if backend is not None:
                try:
                    return (
                        backend.answer_question(image_path, normalized_question, caption, detections),
                        f"openai:{self.settings.openai_model}",
                    )
                except Exception:
                    return None, None

        if provider in {"llava", "llava-next"}:
            backend = self._get_llava_backend()
            if backend is not None:
                try:
                    return backend.answer_question(image_path, normalized_question), f"llava:{self.settings.vlm_model}"
                except Exception:
                    return None, None

        if provider in {"blip", "blip2"}:
            backend = self._get_blip_backend()
            if backend is not None:
                try:
                    return backend.answer_question(image_path, normalized_question), f"blip-vqa:{self.settings.vqa_model}"
                except Exception:
                    return None, None

        if provider == "auto" and self.settings.openai_api_key:
            backend = self._get_openai_backend()
            if backend is not None and (not detections or len(question.split()) > 5):
                try:
                    return (
                        backend.answer_question(image_path, normalized_question, caption, detections),
                        f"openai:{self.settings.openai_model}",
                    )
                except Exception:
                    return None, None

        return None, None

    def _caption_image(self, image_path: str, detections: list[BoundingBox]) -> tuple[str, str]:
        rendered = _render_caption_from_detections(detections)
        if detections:
            return rendered, "detector-caption"

        provider = self.settings.vlm_provider
        if provider == "openai" and self.settings.openai_api_key:
            backend = self._get_openai_backend()
            if backend is not None:
                try:
                    return (
                        backend.answer_question(image_path, "Describe the image in one short sentence.", "", []),
                        f"openai:{self.settings.openai_model}",
                    )
                except Exception:
                    pass

        if provider in {"llava", "llava-next"}:
            backend = self._get_llava_backend()
            if backend is not None:
                try:
                    return backend.answer_question(image_path, "Describe the image in one short sentence."), f"llava:{self.settings.vlm_model}"
                except Exception:
                    pass

        if provider in {"blip", "blip2"}:
            backend = self._get_blip_backend()
            if backend is not None:
                try:
                    return backend.caption(image_path), f"blip-caption:{self.settings.caption_model}"
                except Exception:
                    pass

        return rendered, "fallback-caption"

    def _answer_image_question(
        self,
        image_path: str,
        question: str,
        caption: str,
        detections: list[BoundingBox],
    ) -> tuple[str, str]:
        normalized = question.strip()
        if not normalized:
            return caption, "caption"

        rule_answer, rule_mode = _answer_from_rules(normalized, caption, detections)
        optional_answer, optional_mode = self._answer_with_optional_vlm(image_path, normalized, caption, detections)
        if optional_answer:
            return optional_answer, optional_mode or rule_mode
        return rule_answer, rule_mode

    def _build_visual_proofs(
        self,
        image_path: str,
        question: str,
        caption: str,
        detections: list[BoundingBox],
    ) -> tuple[str, list[ProofItem]]:
        selected_boxes = _select_proof_boxes(question, detections, self.settings.visual_proof_count)
        file_stem = _slugify(Path(image_path).stem) + "-" + uuid.uuid4().hex[:8]
        annotated_path = self.settings.generated_dir / f"{file_stem}-annotated.jpg"
        draw_boxes(image_path, selected_boxes, annotated_path)
        annotated_url = f"/uploads/generated/{annotated_path.name}"

        lead_score = max((box.score for box in selected_boxes[:1]), default=0.38)
        proofs: list[ProofItem] = [
            ProofItem(
                id="P0",
                title="Highlighted uploaded image",
                source_kind="highlighted-image",
                source_id=Path(image_path).name,
                image_url=annotated_url,
                annotated_image_url=annotated_url,
                caption=caption,
                supporting_text=caption,
                score=round(float(lead_score), 4),
                retrieval_channels=["detector", "visual-grounding"],
                explanation=[
                    "This proof shows the uploaded image with the most relevant grounded regions highlighted.",
                    f"The highlighted regions were selected for the question: {question or 'Describe the image.'}",
                ],
                boxes=selected_boxes,
            )
        ]

        if not selected_boxes:
            return annotated_url, proofs

        for index, box in enumerate(selected_boxes, start=1):
            crop_path = self.settings.generated_dir / f"{file_stem}-proof-{index}.jpg"
            save_crop(image_path, box, crop_path)
            crop_url = f"/uploads/generated/{crop_path.name}"
            proofs.append(
                ProofItem(
                    id=f"P{index}",
                    title=f"Detected {box.label}",
                    source_kind="detected-region",
                    source_id=f"{Path(image_path).name}:{index}",
                    image_url=crop_url,
                    annotated_image_url=annotated_url,
                    caption=f"Detected region for {box.label}",
                    supporting_text=f"{box.label} detected with confidence {box.score:.2f}.",
                    score=round(float(box.score), 4),
                    retrieval_channels=["detector"],
                    explanation=[
                        f"Grounded detection localized `{box.label}` as a proof region relevant to the question.",
                        f"Bounding box confidence: {box.score:.2f}.",
                    ],
                    boxes=[box],
                )
            )

        return annotated_url, proofs[: self.settings.visual_proof_count + 1]

    def analyze(self, image_path: str, question: str) -> VisionAnalysisResult:
        detector = self._get_detector()
        detections: list[BoundingBox] = []
        labels = guess_object_labels(question, "")
        if detector is not None:
            try:
                detections = detector.detect(image_path, labels, self.settings.detection_threshold)
            except Exception:
                detections = []
        detections = _prune_detections(detections, self.settings.detection_threshold)

        caption, caption_mode = self._caption_image(image_path, detections)
        answer, answer_mode = self._answer_image_question(image_path, question, caption, detections)
        highlighted_url, proofs = self._build_visual_proofs(image_path, question, caption, detections)

        selected_boxes = [box for proof in proofs[1:] for box in proof.boxes]
        avg_detector_score = sum(box.score for box in selected_boxes[:4]) / max(1, len(selected_boxes[:4]))
        visual_grounding_score = min(
            0.999,
            (0.58 * avg_detector_score) + (0.22 if selected_boxes else 0.0) + (0.2 if answer else 0.0),
        )
        explanation = [
            f"Caption source: {caption_mode}.",
            f"Question answering source: {answer_mode}.",
        ]
        if selected_boxes:
            explanation.append(
                "Relevant grounded objects: "
                + ", ".join(f"{box.label} ({box.score:.2f})" for box in selected_boxes[:4])
                + "."
            )
        else:
            explanation.append("No confident object detections were found, so the answer relies on global image reasoning.")

        return VisionAnalysisResult(
            answer=answer,
            answer_mode=answer_mode,
            caption=caption,
            highlighted_image_url=highlighted_url,
            proofs=proofs,
            visual_grounding_score=round(visual_grounding_score, 3),
            explanation=explanation,
            detected_boxes=detections,
        )
