"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";

type PredictionLabel = "normal" | "benign" | "malignant";

type PredictionResult = {
  label: PredictionLabel;
  confidence: number;
  explanation: string;
  heatmap_url: string;
  probabilities?: Partial<Record<PredictionLabel, number>>;
  processing_notes: string;
  report_draft?: string;
  xai_method?: string;
  model_mode?: string;
};

const labelCopy: Record<PredictionLabel, string> = {
  normal: "Normal appearance",
  benign: "Benign pattern",
  malignant: "Suspicious pattern",
};

const labelTone: Record<PredictionLabel, string> = {
  normal: "status-tag status-tag-normal",
  benign: "status-tag status-tag-benign",
  malignant: "status-tag status-tag-malignant",
};

const confidenceDescriptor = (confidence: number) => {
  if (confidence >= 0.85) {
    return "High";
  }

  if (confidence >= 0.65) {
    return "Moderate";
  }

  return "Low";
};

export default function Home() {
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [clinicianDecision, setClinicianDecision] = useState<"accepted" | "rejected" | "">("");
  const [clinicianComment, setClinicianComment] = useState("");
  const [decisionCaptured, setDecisionCaptured] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">(
    "checking"
  );
  const [backendMode, setBackendMode] = useState("");

  useEffect(() => {
    let revokedUrl = "";

    async function checkBackend() {
      try {
        const response = await fetch("/api/health", { cache: "no-store" });
        const payload = await response.json().catch(() => null);
        setBackendStatus(response.ok ? "online" : "offline");
        setBackendMode(typeof payload?.model_mode === "string" ? payload.model_mode : "");
      } catch {
        setBackendStatus("offline");
        setBackendMode("");
      }
    }

    checkBackend();

    if (uploadedImage) {
      revokedUrl = URL.createObjectURL(uploadedImage);
      setPreviewUrl(revokedUrl);
    } else {
      setPreviewUrl("");
    }

    return () => {
      if (revokedUrl) {
        URL.revokeObjectURL(revokedUrl);
      }
    };
  }, [uploadedImage]);

  const confidencePercentage = useMemo(() => {
    if (!predictionResult) {
      return "--";
    }

    return `${(predictionResult.confidence * 100).toFixed(1)}%`;
  }, [predictionResult]);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setUploadedImage(file);
    setPredictionResult(null);
    setClinicianDecision("");
    setClinicianComment("");
    setDecisionCaptured(false);
    setError("");
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!uploadedImage) {
      setError("Select an ultrasound image before running the analysis.");
      return;
    }

    setLoading(true);
    setError("");
    setPredictionResult(null);
    setDecisionCaptured(false);

    const formData = new FormData();
    formData.append("image", uploadedImage);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      const payload = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(
          payload?.detail ??
            "The analysis service is unavailable. Confirm the FastAPI server is running on port 8000."
        );
      }

      setPredictionResult(payload as PredictionResult);
      setBackendStatus("online");
      setBackendMode(
        payload && typeof payload === "object" && typeof payload.model_mode === "string"
          ? payload.model_mode
          : ""
      );
    } catch (submissionError) {
      setBackendStatus("offline");
      setError(
        submissionError instanceof Error
          ? submissionError.message
          : "The request could not be completed."
      );
    } finally {
      setLoading(false);
    }
  }

  function captureDecision(decision: "accepted" | "rejected") {
    setClinicianDecision(decision);
    setDecisionCaptured(true);
  }

  return (
    <main className="platform-shell">
      <header className="app-header">
        <div className="header-logo">
          <span className="logo-mark" />
          <span className="logo-text">DrCancer</span>
        </div>

        <div className="header-title">Breast ultrasound review workspace</div>

        <div className="header-status">
          <span className={`status-dot status-${backendStatus}`} />
          <span>
            {backendStatus === "checking"
              ? "Checking"
              : backendStatus === "online"
              ? "Online"
              : "Offline"}
          </span>
        </div>
      </header>

      <section className="workspace-grid">
        <section className="main-stage">
          <div className="top-controlbar">
            <div className="meta-strip">
              <div className="meta-item">
                <span className="meta-label">Modality</span>
                <strong>Ultrasound</strong>
              </div>
              <div className="meta-item">
                <span className="meta-label">Study</span>
                <strong>{uploadedImage ? uploadedImage.name : "No file selected"}</strong>
              </div>
              <div className="meta-item">
                <span className="meta-label">Status</span>
                <strong>{predictionResult ? "Processed" : "Ready"}</strong>
              </div>
              <div className="meta-item">
                <span className="meta-label">Inference</span>
                <strong>{backendMode || "Unavailable"}</strong>
              </div>
            </div>

            <form className="action-strip" onSubmit={handleSubmit}>
              <label className="file-field" htmlFor="image-upload">
                <span className="field-label">Select study</span>
                <input
                  id="image-upload"
                  name="image"
                  type="file"
                  accept="image/png,image/jpeg,image/jpg,image/webp"
                  onChange={handleFileChange}
                />
              </label>

              <button className="run-button" type="submit" disabled={loading}>
                {loading ? "Running..." : "Run analysis"}
              </button>
            </form>
          </div>

          {error ? <p className="message error-message">{error}</p> : null}

          <div className="viewer-layout">
            <section className="viewer-panel">
              <div className="section-header">
                <h2>Original ultrasound</h2>
                <span className="panel-state">
                  {predictionResult ? "Image loaded" : "Awaiting upload"}
                </span>
              </div>

              <div className="image-canvas">
                {previewUrl ? (
                  <img src={previewUrl} alt="Uploaded ultrasound preview" className="scan-image" />
                ) : (
                  <div className="canvas-empty">Upload an ultrasound image to begin review.</div>
                )}
              </div>
            </section>

            <aside className="support-rail">
              <section className="support-panel">
                <div className="section-header">
                  <h3>Heatmap</h3>
                </div>

                <div className="mini-canvas">
                  {predictionResult ? (
                    <img
                      src={predictionResult.heatmap_url}
                      alt="Model explanation heatmap"
                      className="scan-image"
                    />
                  ) : (
                    <div className="canvas-empty compact">Available after analysis.</div>
                  )}
                </div>
              </section>

              <section className="support-panel">
                <div className="section-header">
                  <h3>Clinical summary</h3>
                </div>

                {predictionResult ? (
                  <div className="summary-copy">
                    <p>{predictionResult.explanation}</p>
                    {predictionResult.xai_method ? (
                      <p>
                        XAI method: <strong>{predictionResult.xai_method}</strong>
                      </p>
                    ) : null}
                  </div>
                ) : (
                  <div className="canvas-empty compact">Summary will appear here.</div>
                )}
              </section>
            </aside>
          </div>
        </section>

        <aside className="right-rail">
          <section className="side-panel">
            <div className="section-header">
              <h3>Assessment</h3>
            </div>

            {predictionResult ? (
              <div className="assessment-stack">
                <div className="assessment-primary">
                  <span className={labelTone[predictionResult.label]}>
                    {labelCopy[predictionResult.label]}
                  </span>
                  <strong className="score-value">{confidencePercentage}</strong>
                </div>

                <div className="data-row">
                  <span>Confidence</span>
                  <strong>{confidenceDescriptor(predictionResult.confidence)}</strong>
                </div>

                <div className="detail-block">
                  <span className="detail-label">Processing notes</span>
                  <p>{predictionResult.processing_notes}</p>
                </div>

                {predictionResult.probabilities ? (
                  <div className="detail-block">
                    <span className="detail-label">Class probabilities</span>
                    <p>
                      {(["normal", "benign", "malignant"] as PredictionLabel[])
                        .filter((label) => typeof predictionResult.probabilities?.[label] === "number")
                        .map(
                          (label) =>
                            `${labelCopy[label]} ${(
                              (predictionResult.probabilities?.[label] ?? 0) * 100
                            ).toFixed(1)}%`
                        )
                        .join(" • ")}
                    </p>
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="rail-empty">Results appear after analysis.</div>
            )}
          </section>

          <section className="side-panel">
            <div className="section-header">
              <h3>Final review</h3>
            </div>

            <div className="decision-actions">
              <button
                type="button"
                className={`decision-button ${clinicianDecision === "accepted" ? "selected" : ""}`}
                onClick={() => captureDecision("accepted")}
                disabled={!predictionResult}
              >
                Accept result
              </button>
              <button
                type="button"
                className={`decision-button ${clinicianDecision === "rejected" ? "selected" : ""}`}
                onClick={() => captureDecision("rejected")}
                disabled={!predictionResult}
              >
                Reject result
              </button>
            </div>

            <label className="comment-box">
              <span className="field-label">Clinical note</span>
              <textarea
                rows={7}
                placeholder="Add an interpretive note or review comment..."
                value={clinicianComment}
                onChange={(event) => {
                  setClinicianComment(event.target.value);
                  if (predictionResult) {
                    setDecisionCaptured(true);
                  }
                }}
                disabled={!predictionResult}
              />
            </label>

            {predictionResult?.report_draft ? (
              <div className="detail-block">
                <span className="detail-label">AI draft report</span>
                <p>{predictionResult.report_draft}</p>
              </div>
            ) : null}

            {decisionCaptured && predictionResult ? (
              <p className="message success-message">
                Review saved in this session: {clinicianDecision || "comment updated"}.
              </p>
            ) : (
              <p className="support-copy">Clinician action is recorded in the current session.</p>
            )}
          </section>
        </aside>
      </section>
    </main>
  );
}
