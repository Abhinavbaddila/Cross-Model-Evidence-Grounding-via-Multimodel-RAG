const LABEL_ALIASES = {
  people: "person",
  man: "person",
  woman: "person",
  boy: "person",
  girl: "person",
  men: "person",
  women: "person",
  persons: "person",
  players: "person",
  friend: "person",
  friends: "person",
  cellphone: "cell phone",
  mobile: "cell phone",
  phone: "cell phone",
  tvmonitor: "tv",
  diningtable: "dining table",
};

let detectorPromise = null;
let faceDetectorPromise = null;
let ocrWorkerPromise = null;

function normalizeLabel(value) {
  const cleaned = value.trim().toLowerCase();
  return LABEL_ALIASES[cleaned] || cleaned;
}

function tokenize(value) {
  return (value || "")
    .toLowerCase()
    .match(/[a-z0-9]+/g)?.filter(Boolean) || [];
}

function matchedTerms(question, values) {
  const qTokens = new Set(tokenize(question));
  const terms = [];
  for (const value of values) {
    for (const token of tokenize(value)) {
      if (qTokens.has(token) && !terms.includes(token)) {
        terms.push(token);
      }
    }
  }
  return terms.slice(0, 8);
}

function pluralize(label, count) {
  if (count === 1) {
    return label;
  }
  if (label === "person" || label === "face") {
    return "people";
  }
  if (label.endsWith("y")) {
    return `${label.slice(0, -1)}ies`;
  }
  if (label.endsWith("s")) {
    return label;
  }
  return `${label}s`;
}

function parseCountTarget(question) {
  const normalized = question.toLowerCase();
  if (!normalized.includes("how many")) {
    return null;
  }
  const patterns = [
    /how many\s+([a-z ]+?)\s+(?:are|is)\s+(?:there|visible|present)/,
    /how many\s+([a-z ]+?)\s+(?:can you see|do you see)/,
    /how many\s+([a-z ]+?)\s+in\s+the\s+image/,
    /how many\s+([a-z ]+)$/,
  ];

  for (const pattern of patterns) {
    const match = normalized.match(pattern);
    if (match?.[1]) {
      return normalizeLabel(match[1].trim().replace(/\bthe\b/g, "").trim());
    }
  }

  if (/\bpeople\b|\bperson\b|\bfriends\b|\bmen\b|\bwomen\b/.test(normalized)) {
    return "person";
  }
  return null;
}

function describeCounts(counts) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    return "No confident visual objects were detected.";
  }
  const phrases = entries.slice(0, 5).map(([label, count]) => `${count} ${pluralize(label, count)}`);
  if (phrases.length === 1) {
    return `The image shows ${phrases[0]}.`;
  }
  return `The image shows ${phrases.slice(0, -1).join(", ")}, and ${phrases.at(-1)}.`;
}

function looksLikeTextQuestion(question) {
  const normalized = question.toLowerCase();
  return [
    "text",
    "written",
    "write",
    "read",
    "says",
    "say",
    "word",
    "words",
    "number",
    "numbers",
    "phone",
    "date",
    "price",
    "amount",
    "name",
    "email",
    "address",
    "sign",
    "board",
    "poster",
    "banner",
    "notice",
    "label",
  ].some((phrase) => normalized.includes(phrase));
}

function normalizeObjectDetections(detections) {
  return detections
    .filter((item) => item.score >= 0.22)
    .map((item) => ({
      label: normalizeLabel(item.class),
      score: Number(item.score || 0),
      bbox: item.bbox,
      source: "object",
    }));
}

function normalizeFaceDetections(faces) {
  return faces
    .map((item) => {
      const topLeft = Array.isArray(item.topLeft) ? item.topLeft : item.topLeft.arraySync?.() || [0, 0];
      const bottomRight = Array.isArray(item.bottomRight)
        ? item.bottomRight
        : item.bottomRight.arraySync?.() || [0, 0];
      const probability = Array.isArray(item.probability) ? item.probability[0] : item.probability ?? 0.92;
      return {
        label: "face",
        score: Number(probability || 0.92),
        bbox: [
          topLeft[0],
          topLeft[1],
          Math.max(0, bottomRight[0] - topLeft[0]),
          Math.max(0, bottomRight[1] - topLeft[1]),
        ],
        source: "face",
      };
    })
    .filter((item) => item.score >= 0.7);
}

function buildCounts(detections) {
  return detections.reduce((accumulator, item) => {
    const label = normalizeLabel(item.label);
    accumulator[label] = (accumulator[label] || 0) + 1;
    return accumulator;
  }, {});
}

function peopleCountEvidence(objectCounts, faceDetections) {
  const objectCount = objectCounts.person || 0;
  const faceCount = faceDetections.length;
  const chosenCount = Math.max(objectCount, faceCount);

  if (faceCount > objectCount) {
    return {
      count: faceCount,
      basis: "face-detection",
      explanation: `Face detection found ${faceCount} visible people, while body detection found ${objectCount}.`,
      detections: faceDetections,
      confidenceBoost: 0.18,
    };
  }

  return {
    count: chosenCount,
    basis: objectCount > 0 ? "object-detection" : "face-detection",
    explanation:
      objectCount > 0
        ? `Object detection found ${objectCount} people and face detection found ${faceCount}.`
        : `Face detection found ${faceCount} visible people.`,
    detections: objectCount > 0 ? [] : faceDetections,
    confidenceBoost: objectCount > 0 ? 0.14 : 0.16,
  };
}

function getOcrBBox(item) {
  if (!item?.bbox) {
    return null;
  }
  const { x0, y0, x1, y1 } = item.bbox;
  return [x0, y0, Math.max(0, x1 - x0), Math.max(0, y1 - y0)];
}

function normalizeOcrLines(ocrData) {
  const lines = (ocrData?.lines || [])
    .map((line) => ({
      text: (line.text || "").trim(),
      score: Number((line.confidence ?? line.conf ?? 0) / 100),
      bbox: getOcrBBox(line),
    }))
    .filter((line) => line.text);

  const words = (ocrData?.words || [])
    .map((word) => ({
      text: (word.text || "").trim(),
      score: Number((word.confidence ?? word.conf ?? 0) / 100),
      bbox: getOcrBBox(word),
    }))
    .filter((word) => word.text && word.score >= 0.45);

  return { lines, words };
}

function normalizeExtractedText(text) {
  return (text || "").replace(/\s+/g, " ").trim();
}

function scoreTextLine(question, line) {
  const questionTokens = tokenize(question);
  const lineTokens = tokenize(line.text);
  const overlap = lineTokens.filter((token) => questionTokens.includes(token)).length;
  const numericBoost = /\d/.test(line.text) && /\b(number|price|date|phone|amount|year|time)\b/.test(question.toLowerCase()) ? 2 : 0;
  return overlap * 3 + numericBoost + line.score;
}

function extractEntity(question, text) {
  const normalized = question.toLowerCase();
  if (normalized.includes("phone")) {
    return text.match(/(\+?\d[\d\s\-()]{7,}\d)/)?.[1] || null;
  }
  if (normalized.includes("email")) {
    return text.match(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i)?.[0] || null;
  }
  if (normalized.includes("price") || normalized.includes("amount")) {
    return text.match(/(?:₹|\$|Rs\.?)\s?\d[\d,]*(?:\.\d+)?/)?.[0] || null;
  }
  if (normalized.includes("date")) {
    return text.match(/\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b|\b\d{1,2}\s+[A-Za-z]+\s+\d{2,4}\b/)?.[0] || null;
  }
  if (normalized.includes("number")) {
    return text.match(/\b\d[\dA-Za-z\-/.]*\b/)?.[0] || null;
  }
  return null;
}

function answerFromOcr(question, ocr) {
  if (!ocr) {
    return null;
  }

  const fullText = normalizeExtractedText(ocr.fullText);
  if (!fullText) {
    return null;
  }

  const extractedEntity = extractEntity(question, fullText);
  if (extractedEntity) {
    return {
      answer: `The extracted text indicates: ${extractedEntity}.`,
      proofText: fullText,
      summary: "The answer is grounded in OCR text extracted from the uploaded image.",
      matched: matchedTerms(question, [fullText]),
      basis: "ocr-entity",
    };
  }

  const sortedLines = [...ocr.lines].sort((a, b) => scoreTextLine(question, b) - scoreTextLine(question, a));
  const bestLine = sortedLines[0];
  const normalized = question.toLowerCase();

  if (
    normalized.includes("what is written") ||
    normalized.includes("what does") ||
    normalized.includes("read the text") ||
    normalized.includes("what text") ||
    normalized.includes("text in the image")
  ) {
    return {
      answer: `The visible text reads: ${fullText}.`,
      proofText: fullText,
      summary: "The answer is grounded in OCR text extracted from the uploaded image.",
      matched: matchedTerms(question, [fullText]),
      basis: "ocr-full-text",
    };
  }

  if (bestLine && scoreTextLine(question, bestLine) > 0) {
    return {
      answer: `The most relevant extracted text is: ${bestLine.text}.`,
      proofText: bestLine.text,
      summary: "The answer is grounded in the highest-overlap OCR line from the uploaded image.",
      matched: matchedTerms(question, [bestLine.text]),
      basis: "ocr-best-line",
    };
  }

  return {
    answer: `The extracted text is: ${fullText}.`,
    proofText: fullText,
    summary: "The answer is grounded in OCR text extracted from the uploaded image.",
    matched: matchedTerms(question, [fullText]),
    basis: "ocr-fallback",
  };
}

function inferVisualAnswer(question, objectCounts, faceDetections) {
  const normalized = question.trim().toLowerCase();
  const labels = Object.keys(objectCounts);
  const countTarget = parseCountTarget(normalized);

  if (countTarget === "person") {
    const countInfo = peopleCountEvidence(objectCounts, faceDetections);
    if (countInfo.count > 0) {
      return {
        answer: `There ${countInfo.count === 1 ? "is" : "are"} ${countInfo.count} ${pluralize("person", countInfo.count)} visible in the image.`,
        summary: `The count uses ${countInfo.basis.replace("-", " ")} as the stronger signal.`,
        matched: ["people", "person"],
        basis: countInfo.basis,
        countInfo,
      };
    }
  }

  if (countTarget) {
    const count = objectCounts[countTarget] || 0;
    if (count > 0) {
      return {
        answer: `There ${count === 1 ? "is" : "are"} ${count} ${pluralize(countTarget, count)} visible in the image.`,
        summary: `The count is grounded in detected ${pluralize(countTarget, count)}.`,
        matched: [countTarget],
        basis: "object-count",
      };
    }
  }

  if (
    normalized.includes("what is in") ||
    normalized.includes("what's in") ||
    normalized.includes("what objects") ||
    normalized.includes("what can you see") ||
    normalized.includes("identify")
  ) {
    if (!labels.length) {
      return null;
    }
    return {
      answer: `The detected objects include ${labels.slice(0, 6).join(", ")}.`,
      summary: "The answer is grounded in browser-side object detection on the uploaded image.",
      matched: matchedTerms(question, labels),
      basis: "object-list",
    };
  }

  if (
    normalized.startsWith("is there") ||
    normalized.startsWith("are there") ||
    normalized.startsWith("does the image show")
  ) {
    const requested = labels.find((label) => normalized.includes(label));
    if (requested) {
      return {
        answer:
          objectCounts[requested] > 0
            ? `Yes, the image shows ${objectCounts[requested]} ${pluralize(requested, objectCounts[requested])}.`
            : `No, I could not detect ${pluralize(requested, 2)} clearly in the image.`,
        summary: "The answer is grounded in detected objects from the uploaded image.",
        matched: [requested],
        basis: "object-boolean",
      };
    }
  }

  if (normalized.includes("describe")) {
    return {
      answer: describeCounts(objectCounts),
      summary: "The answer is grounded in detected objects from the uploaded image.",
      matched: matchedTerms(question, labels),
      basis: "object-description",
    };
  }

  if (labels.length) {
    return {
      answer: describeCounts(objectCounts),
      summary: "The answer is grounded in detected objects from the uploaded image.",
      matched: matchedTerms(question, labels),
      basis: "object-fallback",
    };
  }

  return null;
}

function confidenceFromEvidence({ objectDetections, faceDetections, ocr, answerBasis }) {
  const objectTop = objectDetections.slice(0, 5);
  const objectAvg = objectTop.length
    ? objectTop.reduce((sum, item) => sum + item.score, 0) / objectTop.length
    : 0;
  const faceAvg = faceDetections.length
    ? faceDetections.reduce((sum, item) => sum + item.score, 0) / faceDetections.length
    : 0;
  const ocrAvg = ocr?.lines?.length
    ? ocr.lines.slice(0, 5).reduce((sum, line) => sum + line.score, 0) / Math.min(ocr.lines.length, 5)
    : 0;

  let confidence = objectAvg * 0.5;
  confidence += faceAvg * 0.22;
  confidence += ocrAvg * 0.24;

  if (answerBasis?.startsWith("ocr")) {
    confidence += 0.12;
  }
  if (answerBasis === "face-detection") {
    confidence += 0.1;
  }
  if (objectDetections.length || faceDetections.length || ocr?.lines?.length) {
    confidence += 0.08;
  }

  return Math.min(0.98, Number(confidence.toFixed(2)));
}

async function loadDetector() {
  if (!detectorPromise) {
    detectorPromise = (async () => {
      const tf = await import("@tensorflow/tfjs");
      await import("@tensorflow/tfjs-backend-webgl");
      await import("@tensorflow/tfjs-backend-cpu");
      try {
        await tf.setBackend("webgl");
      } catch {
        await tf.setBackend("cpu");
      }
      await tf.ready();
      const cocoSsd = await import("@tensorflow-models/coco-ssd");
      return cocoSsd.load({ base: "mobilenet_v2" });
    })();
  }
  return detectorPromise;
}

async function loadFaceDetector() {
  if (!faceDetectorPromise) {
    faceDetectorPromise = (async () => {
      const blazeface = await import("@tensorflow-models/blazeface");
      return blazeface.load();
    })();
  }
  return faceDetectorPromise;
}

async function loadOcrWorker() {
  if (!ocrWorkerPromise) {
    ocrWorkerPromise = (async () => {
      const { createWorker } = await import("tesseract.js");
      const worker = await createWorker("eng");
      return worker;
    })();
  }
  return ocrWorkerPromise;
}

async function runOcr(file) {
  const worker = await loadOcrWorker();
  const result = await worker.recognize(file);
  const { lines, words } = normalizeOcrLines(result?.data);
  return {
    fullText: normalizeExtractedText(result?.data?.text || ""),
    lines,
    words,
  };
}

async function loadImage(file) {
  const objectUrl = URL.createObjectURL(file);
  try {
    const image = await new Promise((resolve, reject) => {
      const element = new Image();
      element.onload = () => resolve(element);
      element.onerror = () => reject(new Error("Could not read the uploaded image."));
      element.src = objectUrl;
    });
    return { image, objectUrl };
  } catch (error) {
    URL.revokeObjectURL(objectUrl);
    throw error;
  }
}

function drawEvidenceImage(image, regions, palette) {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth || image.width;
  canvas.height = image.naturalHeight || image.height;
  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  context.lineWidth = Math.max(3, canvas.width * 0.004);
  context.font = `${Math.max(15, Math.round(canvas.width * 0.024))}px Segoe UI`;

  regions.forEach((region, index) => {
    if (!region?.bbox) {
      return;
    }
    const [x, y, width, height] = region.bbox;
    const color = palette[index % palette.length];
    const label = region.text || region.label || "region";
    context.strokeStyle = color;
    context.fillStyle = color;
    context.strokeRect(x, y, width, height);
    const textWidth = context.measureText(label).width + 14;
    const tagY = Math.max(0, y - 28);
    context.fillRect(x, tagY, textWidth, 24);
    context.fillStyle = "#ffffff";
    context.fillText(label, x + 7, tagY + 17);
  });

  return canvas.toDataURL("image/jpeg", 0.92);
}

function buildProofs({
  image,
  objectUrl,
  question,
  answerInfo,
  objectDetections,
  faceDetections,
  ocr,
  confidence,
}) {
  const proofs = [];

  if (answerInfo?.basis?.startsWith("ocr") && ocr?.lines?.length) {
    const topLines = ocr.lines.slice(0, 6).map((line) => ({ ...line, label: line.text }));
    const annotated = drawEvidenceImage(image, topLines, ["#1d4ed8", "#0f766e", "#9333ea"]);
    proofs.push({
      id: "P0",
      title: "OCR text proof",
      image_url: annotated,
      annotated_image_url: annotated,
      supporting_text: answerInfo.proofText,
      score: confidence,
      retrieval_channels: ["browser-ocr", "uploaded-image"],
      matched_terms: answerInfo.matched,
    });
  } else if (answerInfo?.basis === "face-detection" && faceDetections.length) {
    const faceRegions = faceDetections.slice(0, 16).map((item) => ({
      ...item,
      label: `face ${(item.score * 100).toFixed(0)}%`,
    }));
    const annotated = drawEvidenceImage(image, faceRegions, ["#1d4ed8", "#0f766e"]);
    proofs.push({
      id: "P0",
      title: "People counting proof",
      image_url: annotated,
      annotated_image_url: annotated,
      supporting_text: answerInfo.countInfo?.explanation || "Face detections were used to improve the people count.",
      score: confidence,
      retrieval_channels: ["browser-face-detection", "uploaded-image"],
      matched_terms: answerInfo.matched,
    });
  } else if (objectDetections.length) {
    const objectRegions = objectDetections.slice(0, 10).map((item) => ({
      ...item,
      label: `${item.label} ${(item.score * 100).toFixed(0)}%`,
    }));
    const annotated = drawEvidenceImage(image, objectRegions, ["#1d4ed8", "#0f766e", "#9333ea"]);
    proofs.push({
      id: "P0",
      title: "Detected object proof",
      image_url: annotated,
      annotated_image_url: annotated,
      supporting_text: answerInfo?.summary || describeCounts(buildCounts(objectDetections)),
      score: confidence,
      retrieval_channels: ["browser-object-detection", "uploaded-image"],
      matched_terms: answerInfo?.matched || matchedTerms(question, objectRegions.map((item) => item.label)),
    });
  }

  proofs.push({
    id: proofs.length ? "P1" : "P0",
    title: "Original uploaded image",
    image_url: objectUrl,
    supporting_text:
      answerInfo?.proofText ||
      answerInfo?.summary ||
      "The uploaded image was used as the primary visual evidence for the answer.",
    score: Math.max(0.35, confidence - 0.08),
    retrieval_channels: ["uploaded-image"],
    matched_terms: answerInfo?.matched || [],
  });

  if (ocr?.fullText && !answerInfo?.basis?.startsWith("ocr")) {
    proofs.push({
      id: "P2",
      title: "Extracted text evidence",
      image_url: objectUrl,
      supporting_text: ocr.fullText,
      score: Math.max(0.3, confidence - 0.12),
      retrieval_channels: ["browser-ocr"],
      matched_terms: matchedTerms(question, [ocr.fullText]),
    });
  }

  return proofs;
}

export async function submitDemoQuery({ question, file }) {
  if (!file) {
    return {
      question,
      query_kind: "text",
      answer:
        "This GitHub-hosted demo runs fully in the browser. Upload an image first, then ask a question about that image.",
      answer_mode: "browser-demo",
      confidence: 0.18,
      proof_summary: "No image was uploaded, so the browser demo could not ground an answer.",
      proofs: [],
    };
  }

  const shouldRunOcr = looksLikeTextQuestion(question);
  const [detector, faceDetector] = await Promise.all([loadDetector(), loadFaceDetector()]);
  const { image, objectUrl } = await loadImage(file);

  try {
    const [rawObjects, rawFaces, ocr] = await Promise.all([
      detector.detect(image, 20, 0.22).catch(() => []),
      faceDetector.estimateFaces(image, false).catch(() => []),
      shouldRunOcr ? runOcr(file).catch(() => null) : Promise.resolve(null),
    ]);

    const objectDetections = normalizeObjectDetections(rawObjects);
    const faceDetections = normalizeFaceDetections(rawFaces);
    const objectCounts = buildCounts(objectDetections);

    const ocrAnswer = shouldRunOcr ? answerFromOcr(question, ocr) : null;
    const visualAnswer = inferVisualAnswer(question, objectCounts, faceDetections);
    const answerInfo = ocrAnswer || visualAnswer || {
      answer: describeCounts(objectCounts),
      summary: "The answer is grounded in browser-side visual analysis of the uploaded image.",
      matched: matchedTerms(question, Object.keys(objectCounts)),
      basis: "fallback-visual",
    };

    const confidence = confidenceFromEvidence({
      objectDetections,
      faceDetections,
      ocr,
      answerBasis: answerInfo.basis,
    });

    const proofs = buildProofs({
      image,
      objectUrl,
      question,
      answerInfo,
      objectDetections,
      faceDetections,
      ocr,
      confidence,
    });

    return {
      question,
      query_kind: question?.trim() ? "text+image" : "image",
      answer: answerInfo.answer,
      answer_mode: answerInfo.basis || "browser-vision-demo",
      confidence,
      proof_summary:
        answerInfo.summary ||
        "The answer is grounded in object detections, face detections, or OCR extracted directly in the browser.",
      highlighted_image_url: proofs[0]?.image_url || objectUrl,
      uploaded_file_url: objectUrl,
      uploaded_file_name: file.name,
      proofs,
    };
  } finally {
    // Keep objectUrl alive because it is used by the returned proof cards.
  }
}
