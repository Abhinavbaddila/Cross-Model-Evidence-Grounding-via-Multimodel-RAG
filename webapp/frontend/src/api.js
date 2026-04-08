const explicitBase = import.meta.env.VITE_API_BASE_URL?.trim();

function candidateBases() {
  if (explicitBase) {
    return [explicitBase.replace(/\/$/, "")];
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

  if (window.location.hostname.endsWith("github.io") && !explicitBase) {
    throw new Error(
      "This GitHub Pages site needs a public backend URL. Set VITE_API_BASE_URL during deployment so the frontend can reach the FastAPI API."
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
