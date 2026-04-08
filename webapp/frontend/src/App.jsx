import { useEffect, useState } from "react";
import "./App.css";
import { submitQuery } from "./api";

function formatConfidence(value) {
  return typeof value === "number" ? value.toFixed(2) : null;
}

function ProofCard({ proof, index }) {
  const title = proof.title || proof.caption || `Proof ${index + 1}`;
  const imageUrl = proof.image_url || proof.annotated_image_url || "";
  const supportingText = proof.supporting_text || proof.caption || "";

  return (
    <article className={`proof-card ${index === 0 ? "proof-card-best" : ""}`}>
      {imageUrl ? <img src={imageUrl} alt={title} className="proof-image" /> : null}

      <div className="proof-body">
        <div className="proof-top">
          <span className="proof-badge">{index === 0 ? "Best Proof" : `Proof ${index + 1}`}</span>
          {proof.score ? <span className="proof-score">{proof.score.toFixed(2)}</span> : null}
        </div>

        <h3>{title}</h3>

        {supportingText ? <p className="proof-text">{supportingText}</p> : null}

        {proof.matched_terms?.length ? (
          <div className="chip-row">
            {proof.matched_terms.slice(0, 6).map((term) => (
              <span key={`${proof.id}-${term}`} className="chip">
                {term}
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </article>
  );
}

function App() {
  const [question, setQuestion] = useState("");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }

    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);

  async function handleSubmit(event) {
    event.preventDefault();

    if (!question.trim() && !file) {
      setError("Enter a question or upload an image.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const payload = await submitQuery({ question, file });
      setResponse(payload);
    } catch (err) {
      setError(err.message || "Could not get an answer from the backend.");
    } finally {
      setLoading(false);
    }
  }

  const proofs = response?.proofs ?? response?.evidence ?? [];
  const confidence = formatConfidence(response?.confidence);

  return (
    <div className="app-shell">
      <main className="app">
        <section className="hero-card">
          <div className="hero-copy">
            <p className="eyebrow">Final Review Demo</p>
            <h1>Multimodal Grounded Question Answering</h1>
            <p className="hero-text">Upload an image, ask a question, and review the answer with proof.</p>
          </div>
        </section>

        <section className="panel">
          <form className="query-form" onSubmit={handleSubmit}>
            <label className="field">
              <span className="field-label">Question</span>
              <textarea
                rows="4"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask a question about the image or the indexed corpus"
              />
            </label>

            <div className="form-row">
              <label className="upload-button">
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                />
                Upload Image
              </label>

              <button className="submit-button" type="submit" disabled={loading}>
                {loading ? "Generating..." : "Get Answer"}
              </button>
            </div>

            {file ? (
              <div className="preview-card">
                {previewUrl ? <img src={previewUrl} alt="Uploaded preview" className="preview-image" /> : null}
                <div className="preview-copy">
                  <strong>{file.name}</strong>
                  <span>Ready for query submission</span>
                </div>
                <button type="button" className="remove-button" onClick={() => setFile(null)}>
                  Remove
                </button>
              </div>
            ) : null}

            {error ? <p className="error">{error}</p> : null}
          </form>
        </section>

        {response ? (
          <section className="answer-layout">
            <article className="panel answer-card">
              <div className="answer-head">
                <div>
                  <p className="eyebrow">Answer</p>
                  <h2>Grounded Response</h2>
                </div>

                <div className="meta-row">
                  {confidence ? <span className="meta-chip">Confidence {confidence}</span> : null}
                  {response.query_kind ? <span className="meta-chip">{response.query_kind}</span> : null}
                </div>
              </div>

              <p className="answer-text">{response.answer}</p>
              {response.proof_summary ? <p className="summary-text">{response.proof_summary}</p> : null}
            </article>

            {proofs[0]?.image_url ? (
              <article className="panel spotlight-card">
                <p className="eyebrow">Top Proof</p>
                <img src={proofs[0].image_url} alt={proofs[0].title || "Top proof"} className="spotlight-image" />
              </article>
            ) : null}
          </section>
        ) : (
          <section className="panel empty-card">
            <h2>Ready</h2>
            <p>Submit a question to see the answer and the supporting proofs.</p>
          </section>
        )}

        {proofs.length ? (
          <section className="proof-section">
            <div className="section-head">
              <h2>Proofs</h2>
              <p>Retrieved proof images and supporting text used for the answer.</p>
            </div>

            <div className="proof-grid">
              {proofs.map((proof, index) => (
                <ProofCard key={proof.id || `${index}-${proof.title || "proof"}`} proof={proof} index={index} />
              ))}
            </div>
          </section>
        ) : null}
      </main>
    </div>
  );
}

export default App;
