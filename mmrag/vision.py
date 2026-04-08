from __future__ import annotations

import base64
import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import torch
from PIL import Image, ImageDraw

from .config import Settings
from .corpus import salient_terms
from .schemas import BoundingBox, ProofItem

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:  # pragma: no cover - optional dependency
    RapidOCR = None


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
    "sign",
    "poster",
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
    "text": "text",
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

TEXT_QUERY_HINTS = {
    "text",
    "written",
    "read",
    "write",
    "says",
    "word",
    "words",
    "number",
    "numbers",
    "price",
    "amount",
    "date",
    "time",
    "phone",
    "email",
    "address",
    "room",
    "invoice",
    "bill",
    "poster",
    "sign",
    "board",
    "banner",
    "label",
    "notice",
}


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "item"


def _pluralize(label: str, count: int) -> str:
    if count == 1:
        return label
    if label == "person":
        return "people"
    if label.endswith("y") and not label.endswith("ay"):
        return label[:-1] + "ies"
    if label.endswith(("s", "x", "z", "ch", "sh")):
        return label + "es"
    return label + "s"


def _normalize_label(label: str) -> str:
    normalized = re.sub(r"\s+", " ", label.strip().lower())
    return LABEL_ALIASES.get(normalized, normalized)


def _shorten_label(value: str, limit: int = 28) -> str:
    compact = re.sub(r"\s+", " ", value.strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _plural_forms(label: str) -> set[str]:
    forms = {label}
    if label == "person":
        forms.update({"people", "persons"})
    elif label.endswith("y") and not label.endswith("ay"):
        forms.add(label[:-1] + "ies")
    elif label.endswith(("s", "x", "z", "ch", "sh")):
        forms.add(label + "es")
    else:
        forms.add(label + "s")
    return forms


def _question_mentions_label(question: str, label: str) -> bool:
    normalized = question.lower()
    return any(re.search(rf"\b{re.escape(form)}\b", normalized) for form in _plural_forms(label))


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


def _looks_like_text_question(question: str) -> bool:
    normalized = question.lower()
    return any(hint in normalized for hint in TEXT_QUERY_HINTS)


def _tokenize_text(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


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
            label = _shorten_label(f"{box.label} {box.score:.2f}")
            anchor_y = max(0, box.y1 - 24)
            text_bbox = draw.textbbox((box.x1 + 6, anchor_y + 2), label)
            label_width = max(80, int(text_bbox[2] - text_bbox[0]) + 12)
            label_width = min(label_width, image.width - int(box.x1) - 4)
            draw.rectangle((box.x1, anchor_y, box.x1 + label_width, anchor_y + 22), fill=outline)
            draw.text((box.x1 + 6, anchor_y + 2), label, fill="white")
        image.save(output_path)


def save_crop(image_path: str, box: BoundingBox, output_path: Path) -> None:
    with Image.open(image_path).convert("RGB") as image:
        x1 = max(0, int(math.floor(box.x1)))
        y1 = max(0, int(math.floor(box.y1)))
        x2 = min(image.width, int(math.ceil(box.x2)))
        y2 = min(image.height, int(math.ceil(box.y2)))
        image.crop((x1, y1, x2, y2)).save(output_path)


def best_count_target(question: str, detections: list[BoundingBox]) -> str | None:
    normalized = question.lower()
    if "how many" not in normalized:
        return None
    for label in COMMON_OBJECT_LABELS:
        if _question_mentions_label(normalized, label):
            canonical = _normalize_label(label)
            if canonical in {"people", "man", "woman", "boy", "girl", "player"}:
                return "person"
            return canonical
    if detections:
        return _normalize_label(detections[0].label)
    if any(word in normalized for word in ["people", "person", "friends", "men", "women"]):
        return "person"
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
    labels = {_normalize_label(label) for label in COMMON_OBJECT_LABELS if label in question.lower()}
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
    return [box for box in detections if box.score >= dynamic_threshold][:16]


def _question_terms(question: str) -> set[str]:
    return set(_tokenize_text(question))


def _ocr_line_score(question: str, text: str, score: float) -> float:
    terms = _question_terms(question)
    line_terms = set(_tokenize_text(text))
    overlap = len(terms.intersection(line_terms))
    numeric_boost = 0.0
    if re.search(r"\d", text) and any(term in terms for term in {"number", "price", "amount", "date", "time", "phone", "room"}):
        numeric_boost = 1.2
    return overlap * 2.5 + numeric_boost + score


def _extract_text_entity(question: str, text: str) -> str | None:
    normalized = question.lower()
    patterns: list[str] = []
    if any(term in normalized for term in ["phone", "mobile", "contact"]):
        patterns.append(r"(\+?\d[\d\s\-()]{7,}\d)")
    if "email" in normalized:
        patterns.append(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}")
    if any(term in normalized for term in ["price", "amount", "total", "bill"]):
        patterns.append(r"(?:₹|\$|Rs\.?)\s?\d[\d,]*(?:\.\d+)?")
    if "date" in normalized:
        patterns.append(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b|\b\d{1,2}\s+[A-Za-z]+\s+\d{2,4}\b")
    if "time" in normalized:
        patterns.append(r"\b\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)?\b")
    if "room" in normalized:
        patterns.append(r"\b(?:room|hall|gate)\s*[:#-]?\s*[A-Za-z0-9-]+\b")
    if "invoice" in normalized:
        patterns.append(r"\b(?:invoice|bill)\s*(?:no|number)?\s*[:#-]?\s*[A-Za-z0-9-]+\b")
    if "number" in normalized:
        patterns.append(r"\b[A-Z0-9-]{3,}\b")
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


@dataclass(slots=True)
class OCRLine:
    text: str
    score: float
    box: BoundingBox


@dataclass(slots=True)
class OCRResult:
    full_text: str
    lines: list[OCRLine]


@dataclass(slots=True)
class CountEvidence:
    count: int
    basis: str
    boxes: list[BoundingBox]
    explanation: str


@dataclass(slots=True)
class AnswerDecision:
    answer: str
    mode: str
    explanation: str
    proof_boxes: list[BoundingBox] = field(default_factory=list)
    proof_lines: list[OCRLine] = field(default_factory=list)


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
            max_det=32,
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


class HaarFaceDetector:
    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(str(cascade_path))
        if self.classifier.empty():
            raise RuntimeError("OpenCV haar cascade for frontal face detection is unavailable")

    def detect(self, image_path: str) -> list[BoundingBox]:
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        min_size = max(24, min(gray.shape[:2]) // 14)
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(min_size, min_size),
        )
        return [
            BoundingBox(
                label="face",
                score=0.9,
                x1=float(x),
                y1=float(y),
                x2=float(x + w),
                y2=float(y + h),
            )
            for x, y, w, h in faces
        ][:24]


class RapidOCRBackend:
    def __init__(self) -> None:
        if RapidOCR is None:
            raise RuntimeError("rapidocr_onnxruntime is not installed")
        self.engine = RapidOCR()

    def read(self, image_path: str) -> OCRResult:
        result, _ = self.engine(image_path)
        lines: list[OCRLine] = []
        full_text_parts: list[str] = []
        for item in result or []:
            points, text, score = item
            text = re.sub(r"\s+", " ", str(text)).strip()
            if not text:
                continue
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            box = BoundingBox(
                label=_shorten_label(text),
                score=float(score),
                x1=min(xs),
                y1=min(ys),
                x2=max(xs),
                y2=max(ys),
            )
            lines.append(OCRLine(text=text, score=float(score), box=box))
            full_text_parts.append(text)
        lines.sort(key=lambda line: (line.box.y1, line.box.x1))
        return OCRResult(full_text=" ".join(full_text_parts).strip(), lines=lines)


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
        self._face_detector: HaarFaceDetector | None = None
        self._face_detector_failed = False
        self._ocr_backend: RapidOCRBackend | None = None
        self._ocr_failed = False

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

    def _get_face_detector(self) -> HaarFaceDetector | None:
        if self._face_detector_failed:
            return None
        if self._face_detector is None:
            try:
                self._face_detector = HaarFaceDetector()
            except Exception:
                self._face_detector_failed = True
                self._face_detector = None
        return self._face_detector

    def _get_ocr_backend(self) -> RapidOCRBackend | None:
        if self._ocr_failed:
            return None
        if self._ocr_backend is None:
            try:
                self._ocr_backend = RapidOCRBackend()
            except Exception:
                self._ocr_failed = True
                self._ocr_backend = None
        return self._ocr_backend

    def _should_try_ocr(self, question: str, detections: list[BoundingBox]) -> bool:
        if _looks_like_text_question(question):
            return True
        if not detections:
            return True
        labels = {_normalize_label(box.label) for box in detections[:6]}
        return bool(labels.intersection({"book", "document", "page", "chart", "sign", "poster", "tv"}))

    def _person_count_evidence(self, detections: list[BoundingBox], face_boxes: list[BoundingBox]) -> CountEvidence:
        person_boxes = [box for box in detections if _normalize_label(box.label) == "person"]
        face_count = len(face_boxes)
        person_count = len(person_boxes)
        if face_count >= person_count + 2 and face_count > 0:
            return CountEvidence(
                count=face_count,
                basis="face-count",
                boxes=face_boxes,
                explanation=f"Face detection found {face_count} visible people while object detection found {person_count}.",
            )
        if person_count > 0:
            return CountEvidence(
                count=person_count,
                basis="person-detection",
                boxes=person_boxes,
                explanation=f"Object detection found {person_count} people and face detection found {face_count}.",
            )
        return CountEvidence(
            count=face_count,
            basis="face-count",
            boxes=face_boxes,
            explanation=f"Face detection found {face_count} visible people.",
        )

    def _answer_from_ocr(self, question: str, ocr_result: OCRResult | None) -> AnswerDecision | None:
        if ocr_result is None or not ocr_result.full_text:
            return None

        entity = _extract_text_entity(question, ocr_result.full_text)
        if entity:
            matching_lines = [line for line in ocr_result.lines if entity.lower() in line.text.lower()]
            return AnswerDecision(
                answer=f"The extracted text indicates: {entity}.",
                mode="ocr-entity",
                explanation="OCR extracted a direct entity match for the question.",
                proof_lines=matching_lines or ocr_result.lines[:2],
            )

        if any(phrase in question.lower() for phrase in ["what is written", "what does", "read the text", "what text"]):
            return AnswerDecision(
                answer=f"The visible text reads: {ocr_result.full_text}.",
                mode="ocr-read",
                explanation="OCR extracted the visible text directly from the uploaded image.",
                proof_lines=ocr_result.lines[: min(4, len(ocr_result.lines))],
            )

        scored_lines = sorted(
            ocr_result.lines,
            key=lambda line: _ocr_line_score(question, line.text, line.score),
            reverse=True,
        )
        if scored_lines and _ocr_line_score(question, scored_lines[0].text, scored_lines[0].score) > 0.9:
            lead = scored_lines[0]
            return AnswerDecision(
                answer=f"The most relevant extracted text is: {lead.text}.",
                mode="ocr-best-line",
                explanation="OCR found a line with strong overlap to the question.",
                proof_lines=scored_lines[: min(3, len(scored_lines))],
            )

        return None

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

    def _caption_image(
        self,
        image_path: str,
        detections: list[BoundingBox],
        ocr_result: OCRResult | None = None,
    ) -> tuple[str, str]:
        rendered = _render_caption_from_detections(detections)
        if detections:
            return rendered, "detector-caption"
        if ocr_result is not None and ocr_result.full_text:
            preview = _shorten_label(ocr_result.full_text, limit=72)
            return f"The image contains readable text: {preview}.", "ocr-caption"

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
        face_boxes: list[BoundingBox],
        ocr_result: OCRResult | None,
    ) -> AnswerDecision:
        normalized = question.strip()
        if not normalized:
            return AnswerDecision(
                answer=caption,
                mode="caption",
                explanation="The answer falls back to the grounded image caption.",
                proof_boxes=_select_proof_boxes(question, detections, self.settings.visual_proof_count),
            )

        if _looks_like_text_question(normalized):
            ocr_answer = self._answer_from_ocr(normalized, ocr_result)
            if ocr_answer is not None:
                return ocr_answer

        count_target = best_count_target(normalized, detections)
        if count_target == "person":
            evidence = self._person_count_evidence(detections, face_boxes)
            if evidence.count:
                return AnswerDecision(
                    answer=f"There are {evidence.count} people visible in the image.",
                    mode=evidence.basis,
                    explanation=evidence.explanation,
                    proof_boxes=evidence.boxes[: self.settings.visual_proof_count + 4],
                )

        rule_answer, rule_mode = _answer_from_rules(normalized, caption, detections)
        optional_answer, optional_mode = self._answer_with_optional_vlm(image_path, normalized, caption, detections)
        if optional_answer:
            return AnswerDecision(
                answer=optional_answer,
                mode=optional_mode or rule_mode,
                explanation="A vision-language model refined the grounded answer using the uploaded image.",
                proof_boxes=_select_proof_boxes(question, detections, self.settings.visual_proof_count),
            )
        return AnswerDecision(
            answer=rule_answer,
            mode=rule_mode,
            explanation="Rule-based grounded reasoning selected the answer from detected evidence.",
            proof_boxes=_select_proof_boxes(question, detections, self.settings.visual_proof_count),
        )

    def _build_detection_proofs(
        self,
        image_path: str,
        question: str,
        caption: str,
        detections: list[BoundingBox],
        proof_boxes: list[BoundingBox] | None = None,
        *,
        proof_title: str = "Highlighted uploaded image",
        lead_channel: str = "detector",
        lead_supporting_text: str | None = None,
    ) -> tuple[str, list[ProofItem]]:
        selected_boxes = proof_boxes or _select_proof_boxes(question, detections, self.settings.visual_proof_count)
        file_stem = _slugify(Path(image_path).stem) + "-" + uuid.uuid4().hex[:8]
        annotated_path = self.settings.generated_dir / f"{file_stem}-annotated.jpg"
        draw_boxes(image_path, selected_boxes, annotated_path)
        annotated_url = f"/uploads/generated/{annotated_path.name}"
        lead_score = max((box.score for box in selected_boxes[:1]), default=0.38)
        proofs: list[ProofItem] = [
            ProofItem(
                id="P0",
                title=proof_title,
                source_kind="highlighted-image",
                source_id=Path(image_path).name,
                image_url=annotated_url,
                annotated_image_url=annotated_url,
                caption=caption,
                supporting_text=lead_supporting_text or caption,
                score=round(float(lead_score), 4),
                retrieval_channels=[lead_channel, "visual-grounding"],
                explanation=[
                    "This proof shows the uploaded image with the most relevant grounded regions highlighted.",
                    f"The highlighted regions were selected for the question: {question or 'Describe the image.'}",
                ],
                boxes=selected_boxes,
            )
        ]
        if not selected_boxes:
            return annotated_url, proofs

        for index, box in enumerate(selected_boxes[: self.settings.visual_proof_count], start=1):
            crop_path = self.settings.generated_dir / f"{file_stem}-proof-{index}.jpg"
            save_crop(image_path, box, crop_path)
            crop_url = f"/uploads/generated/{crop_path.name}"
            proofs.append(
                ProofItem(
                    id=f"P{index}",
                    title=f"Detected {_shorten_label(box.label, 24)}",
                    source_kind="detected-region",
                    source_id=f"{Path(image_path).name}:{index}",
                    image_url=crop_url,
                    annotated_image_url=annotated_url,
                    caption=f"Detected region for {box.label}",
                    supporting_text=f"{box.label} detected with confidence {box.score:.2f}.",
                    score=round(float(box.score), 4),
                    retrieval_channels=[lead_channel],
                    explanation=[
                        f"Grounded evidence localized `{box.label}` as a region relevant to the question.",
                        f"Bounding box confidence: {box.score:.2f}.",
                    ],
                    boxes=[box],
                )
            )

        return annotated_url, proofs[: self.settings.visual_proof_count + 1]

    def _build_ocr_proofs(
        self,
        image_path: str,
        question: str,
        caption: str,
        lines: list[OCRLine],
    ) -> tuple[str, list[ProofItem]]:
        ocr_boxes = [
            BoundingBox(
                label=_shorten_label(line.text, 30),
                score=line.score,
                x1=line.box.x1,
                y1=line.box.y1,
                x2=line.box.x2,
                y2=line.box.y2,
            )
            for line in lines
        ]
        annotated_url, proofs = self._build_detection_proofs(
            image_path,
            question,
            caption,
            ocr_boxes,
            proof_boxes=ocr_boxes[: self.settings.visual_proof_count],
            proof_title="OCR highlighted text",
            lead_channel="ocr",
            lead_supporting_text=" ".join(line.text for line in lines[:3]),
        )
        for proof, line in zip(proofs[1:], lines[: self.settings.visual_proof_count]):
            proof.title = "OCR text region"
            proof.caption = f"Extracted text: {line.text}"
            proof.supporting_text = line.text
            proof.explanation = [
                "OCR extracted this text region from the uploaded image.",
                f"Recognition confidence: {line.score:.2f}.",
            ]
        return annotated_url, proofs

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

        face_boxes: list[BoundingBox] = []
        face_detector = self._get_face_detector()
        if face_detector is not None:
            try:
                face_boxes = face_detector.detect(image_path)
            except Exception:
                face_boxes = []

        ocr_result: OCRResult | None = None
        if self._should_try_ocr(question, detections):
            ocr_backend = self._get_ocr_backend()
            if ocr_backend is not None:
                try:
                    ocr_result = ocr_backend.read(image_path)
                except Exception:
                    ocr_result = None

        caption, caption_mode = self._caption_image(image_path, detections, ocr_result=ocr_result)
        decision = self._answer_image_question(image_path, question, caption, detections, face_boxes, ocr_result)

        if decision.proof_lines:
            highlighted_url, proofs = self._build_ocr_proofs(image_path, question, caption, decision.proof_lines)
        elif decision.mode == "face-count" and decision.proof_boxes:
            highlighted_url, proofs = self._build_detection_proofs(
                image_path,
                question,
                caption,
                decision.proof_boxes,
                proof_boxes=decision.proof_boxes,
                proof_title="Counted faces in uploaded image",
                lead_channel="face-detector",
                lead_supporting_text=decision.explanation,
            )
        else:
            highlighted_url, proofs = self._build_detection_proofs(
                image_path,
                question,
                caption,
                detections,
                proof_boxes=decision.proof_boxes,
                proof_title="Highlighted uploaded image",
                lead_channel="detector",
                lead_supporting_text=decision.explanation,
            )

        selected_boxes = [box for proof in proofs for box in proof.boxes]
        avg_selected_score = sum(box.score for box in selected_boxes[:6]) / max(1, len(selected_boxes[:6]))
        ocr_signal = 0.0
        if ocr_result is not None and ocr_result.lines:
            ocr_signal = sum(line.score for line in ocr_result.lines[:4]) / max(1, len(ocr_result.lines[:4]))
        visual_grounding_score = min(
            0.999,
            (0.5 * avg_selected_score)
            + (0.18 if selected_boxes else 0.0)
            + (0.16 if ocr_signal else 0.0)
            + (0.16 if decision.answer else 0.0),
        )
        explanation = [
            f"Caption source: {caption_mode}.",
            f"Question answering source: {decision.mode}.",
            decision.explanation,
        ]
        if selected_boxes:
            explanation.append(
                "Relevant grounded regions: "
                + ", ".join(f"{box.label} ({box.score:.2f})" for box in selected_boxes[:4])
                + "."
            )
        if ocr_result is not None and ocr_result.full_text:
            explanation.append(f"OCR extracted text: {_shorten_label(ocr_result.full_text, 96)}.")
        if face_boxes:
            explanation.append(f"Face detector found {len(face_boxes)} visible faces.")
        if not selected_boxes and not ocr_result:
            explanation.append("No confident detections were found, so the answer relies on fallback image reasoning.")

        return VisionAnalysisResult(
            answer=decision.answer,
            answer_mode=decision.mode,
            caption=caption,
            highlighted_image_url=highlighted_url,
            proofs=proofs,
            visual_grounding_score=round(visual_grounding_score, 3),
            explanation=explanation,
            detected_boxes=detections,
        )
