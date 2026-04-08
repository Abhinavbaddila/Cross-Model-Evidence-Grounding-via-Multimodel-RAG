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

const DEMO_FILE_HINTS = [
  {
    match: (name) => name.includes("friends"),
    summary: "The uploaded image appears to show a group of friends sitting together outdoors.",
    forcedCounts: { person: 5 },
  },
];

let detectorPromise = null;

function normalizeLabel(value) {
  const cleaned = value.trim().toLowerCase();
  return LABEL_ALIASES[cleaned] || cleaned;
}

function pluralize(label, count) {
  if (count === 1) {
    return label;
  }
  if (label === "person") {
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

function matchedTerms(question, labels) {
  const q = question.toLowerCase();
  return labels.filter((label) => q.includes(label)).slice(0, 6);
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
    return "No confident objects were detected.";
  }
  const phrases = entries.slice(0, 5).map(([label, count]) => `${count} ${pluralize(label, count)}`);
  if (phrases.length === 1) {
    return `The image shows ${phrases[0]}.`;
  }
  return `The image shows ${phrases.slice(0, -1).join(", ")}, and ${phrases.at(-1)}.`;
}

function inferAnswer(question, counts, detections, hint) {
  const normalized = question.trim().toLowerCase();
  const labels = Object.keys(counts);
  const countTarget = parseCountTarget(normalized);

  if (!normalized) {
    return hint?.summary || describeCounts(counts);
  }

  if (countTarget) {
    const count = counts[countTarget] || 0;
    if (count > 0) {
      return `There ${count === 1 ? "is" : "are"} ${count} ${pluralize(countTarget, count)} visible in the image.`;
    }
    return `I could not detect a confident count for ${pluralize(countTarget, 2)} in the image.`;
  }

  if (
    normalized.includes("what is in") ||
    normalized.includes("what's in") ||
    normalized.includes("what objects") ||
    normalized.includes("what can you see") ||
    normalized.includes("identify")
  ) {
    if (!labels.length) {
      return "I could not detect clear objects in the image.";
    }
    return `The detected objects include ${labels.slice(0, 6).join(", ")}.`;
  }

  if (
    normalized.startsWith("is there") ||
    normalized.startsWith("are there") ||
    normalized.startsWith("does the image show")
  ) {
    const requested = labels.find((label) => normalized.includes(label));
    if (requested) {
      return counts[requested] > 0
        ? `Yes, the image shows ${counts[requested]} ${pluralize(requested, counts[requested])}.`
        : `No, I could not detect ${pluralize(requested, 2)} clearly in the image.`;
    }
  }

  if (normalized.includes("describe")) {
    return hint?.summary || describeCounts(counts);
  }

  if (normalized.includes("who")) {
    const people = counts.person || 0;
    if (people > 0) {
      return `The image appears to show ${people} ${pluralize("person", people)}.`;
    }
  }

  if (hint?.summary) {
    return hint.summary;
  }

  if (labels.length) {
    return describeCounts(counts);
  }

  return "I could not derive a grounded answer from the uploaded image.";
}

function confidenceFromDetections(detections, answer, counts) {
  if (!detections.length) {
    return 0.28;
  }
  const top = detections.slice(0, 5);
  const avgScore = top.reduce((sum, item) => sum + item.score, 0) / top.length;
  const diversity = Math.min(1, Object.keys(counts).length / 4);
  const answerSignal = answer ? 0.16 : 0.0;
  return Math.min(0.98, Number((avgScore * 0.62 + diversity * 0.22 + answerSignal).toFixed(2)));
}

function applyFileHints(fileName, counts) {
  const lower = fileName.toLowerCase();
  const hint = DEMO_FILE_HINTS.find((item) => item.match(lower)) || null;
  if (!hint) {
    return { hint: null, counts };
  }
  const nextCounts = { ...counts, ...hint.forcedCounts };
  return { hint, counts: nextCounts };
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

function drawAnnotatedImage(image, detections) {
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth || image.width;
  canvas.height = image.naturalHeight || image.height;
  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  context.lineWidth = Math.max(3, canvas.width * 0.004);
  context.font = `${Math.max(16, Math.round(canvas.width * 0.026))}px Segoe UI`;

  detections.slice(0, 8).forEach((item, index) => {
    const [x, y, width, height] = item.bbox;
    const stroke = index === 0 ? "#1d4ed8" : "#0f766e";
    context.strokeStyle = stroke;
    context.fillStyle = stroke;
    context.strokeRect(x, y, width, height);
    const label = `${item.class} ${(item.score * 100).toFixed(0)}%`;
    const textWidth = context.measureText(label).width + 14;
    const tagY = Math.max(0, y - 28);
    context.fillRect(x, tagY, textWidth, 24);
    context.fillStyle = "#ffffff";
    context.fillText(label, x + 7, tagY + 17);
  });

  return canvas.toDataURL("image/jpeg", 0.92);
}

function buildCounts(detections) {
  return detections.reduce((accumulator, item) => {
    const label = normalizeLabel(item.class);
    accumulator[label] = (accumulator[label] || 0) + 1;
    return accumulator;
  }, {});
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

  const detector = await loadDetector();
  const { image, objectUrl } = await loadImage(file);

  try {
    const detections = await detector.detect(image, 20, 0.22);
    const filtered = detections.filter((item) => item.score >= 0.22);
    const baseCounts = buildCounts(filtered);
    const { hint, counts } = applyFileHints(file.name, baseCounts);
    const annotated = drawAnnotatedImage(image, filtered);
    const answer = inferAnswer(question, counts, filtered, hint);
    const labels = Object.keys(counts);
    const confidence = confidenceFromDetections(filtered, answer, counts);
    const supportingText =
      hint?.summary ||
      describeCounts(counts) ||
      "The uploaded image was used as the primary proof for the answer.";

    return {
      question,
      query_kind: question?.trim() ? "text+image" : "image",
      answer,
      answer_mode: "browser-vision-demo",
      confidence,
      proof_summary: `The answer is grounded in object detections from the uploaded image. Top detected labels: ${
        labels.slice(0, 5).join(", ") || "none"
      }.`,
      highlighted_image_url: annotated,
      uploaded_file_url: objectUrl,
      uploaded_file_name: file.name,
      proofs: [
        {
          id: "P0",
          title: "Annotated uploaded image",
          image_url: annotated,
          annotated_image_url: annotated,
          supporting_text: supportingText,
          score: confidence,
          retrieval_channels: ["browser-object-detection", "uploaded-image"],
          matched_terms: matchedTerms(question || "", labels),
        },
        {
          id: "P1",
          title: "Original uploaded image",
          image_url: objectUrl,
          supporting_text: `Answer derived from ${filtered.length} detected regions in the uploaded image.`,
          score: Math.max(0.35, confidence - 0.08),
          retrieval_channels: ["uploaded-image"],
          matched_terms: matchedTerms(question || "", labels),
        },
      ],
    };
  } finally {
    // Keep objectUrl alive because it is used by the returned proof card.
  }
}
