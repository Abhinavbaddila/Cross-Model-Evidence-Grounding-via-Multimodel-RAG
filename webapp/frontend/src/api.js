const explicitBase = import.meta.env.VITE_API_BASE_URL?.trim();
const STORAGE_KEY = "mmrag_api_base_url";

function normalizeBaseUrl(value) {
  return value?.trim().replace(/\/$/, "") || "";
}

function readStoredBase() {
  if (typeof window === "undefined") {
    return "";
  }
  return normalizeBaseUrl(window.localStorage.getItem(STORAGE_KEY) || "");
}

export function saveApiBaseUrl(value) {
  if (typeof window === "undefined") {
    return;
  }
  const normalized = normalizeBaseUrl(value);
  if (normalized) {
    window.localStorage.setItem(STORAGE_KEY, normalized);
  } else {
    window.localStorage.removeItem(STORAGE_KEY);
  }
}

export function getConfiguredApiBaseUrl() {
  return normalizeBaseUrl(explicitBase) || readStoredBase();
}

export function isHostedFrontend() {
  return typeof window !== "undefined" && window.location.hostname.endsWith("github.io");
}

function candidateBases() {
  const configured = getConfiguredApiBaseUrl();
  if (configured) {
    return [configured];
  }

  const host = window.location.hostname;
  if (host.endsWith("github.io")) {
    return [];
  }

  const bases = [""];
  if (host === "127.0.0.1" || host === "localhost") {
    bases.push("http://127.0.0.1:9000");
    bases.push("http://localhost:9000");
  }
  return [...new Set(bases)];
}

async function readPayload(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response.text();
}

async function request(path, options = {}) {
  let lastError = null;

  for (const base of candidateBases()) {
    try {
      const response = await fetch(`${base}${path}`, options);
      const payload = await readPayload(response);
      if (!response.ok) {
        const detail =
          typeof payload === "string"
            ? payload
            : payload?.detail || payload?.message || JSON.stringify(payload);
        throw new Error(detail || `Request failed with status ${response.status}`);
      }
      return payload;
    } catch (error) {
      lastError = error;
    }
  }

  if (window.location.hostname.endsWith("github.io") && !getConfiguredApiBaseUrl()) {
    throw new Error(
      "Paste a public backend URL first. Use your active trycloudflare URL, then submit again."
    );
  }

  throw new Error(
    lastError?.message ||
      "Could not reach the backend. Make sure the FastAPI server is running on http://127.0.0.1:9000."
  );
}

export function fetchStatus() {
  return request("/api/status");
}

export function submitQuery({ question, file }) {
  const form = new FormData();
  form.append("question", question ?? "");
  if (file) {
    form.append("file", file);
  }

  return request("/api/query", {
    method: "POST",
    body: form,
  });
}
