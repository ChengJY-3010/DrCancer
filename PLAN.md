# 2-Hour Demo Implementation Plan: Breast Ultrasound AI Review Flow

## Summary
Build a demo-only system using `Next.js` for the web dashboard and `FastAPI` for the backend. The flow will support image upload, backend preprocessing, simulated AI classification, simulated Grad-CAM heatmap generation, a short clinical-style explanation, and a doctor review screen with accept/reject/comment actions.

The goal is not medical accuracy. The goal is a stable, believable end-to-end demo that matches the workflow you described and can be completed within 2 hours.

## Key Changes
### Architecture
- Frontend: `Next.js` app with one main doctor dashboard page.
- Backend: `FastAPI` service with a single inference endpoint and optional health endpoint.
- Storage: keep everything in local temp/static folders only for the active demo session.
- AI layer: implement a hybrid demo pipeline.
  - Real preprocessing on uploaded image.
  - Mocked or rule-based classifier returning `normal | benign | malignant` plus confidence.
  - Fake Grad-CAM heatmap generated as an overlay image to simulate attention regions.
  - Template-based report generator that converts class + confidence into clinician-friendly text.

### Backend behavior
- Add `POST /predict` endpoint:
  - Accept multipart image upload.
  - Resize image to model input size, normalize, convert to RGB/array/tensor-ready format.
  - Run demo classifier function.
  - Generate heatmap image file or base64 string.
  - Generate explanation text from prediction result.
  - Return JSON:
    - `label`
    - `confidence`
    - `heatmap_url` or `heatmap_base64`
    - `explanation`
    - `processing_notes` for demo transparency
- Add `GET /health` endpoint for quick startup verification.
- Keep classifier logic isolated behind a function/module so a real CNN can replace it later without changing the API.

### Frontend behavior
- Create one dashboard page with these sections:
  - Upload panel
  - Original ultrasound preview
  - Prediction card
  - Confidence display
  - Heatmap preview
  - Clinical-style explanation
  - Clinician decision controls: `Accept`, `Reject`, `Comment`
- On upload:
  - preview the selected image immediately
  - send it to backend
  - show loading state
  - render backend results when complete
- Decision controls:
  - no database needed
  - store the clinician action and comment in frontend state only
  - show a confirmation banner like “Decision captured for demo”

### Report generator rules
- Use fixed templates keyed by prediction label and confidence band.
- Confidence bands:
  - high: `>= 0.85`
  - medium: `0.65 - 0.84`
  - low: `< 0.65`
- Example behavior:
  - `malignant + high` -> “Suspicious features are highlighted; further diagnostic review is advised.”
  - `benign + medium` -> “Features appear less suspicious, but clinical correlation is recommended.”
  - `normal + high` -> “No clearly suspicious region is emphasized in this image.”
- Include one short disclaimer in UI: `Demo output only. Not for clinical use.`

## Public Interfaces
### Backend API
- `POST /predict`
  - Request: multipart form with field `image`
  - Response JSON:
```json
{
  "label": "benign",
  "confidence": 0.82,
  "heatmap_url": "/outputs/heatmap-123.png",
  "explanation": "The highlighted region appears less suspicious, which is more consistent with a benign pattern in this demo.",
  "processing_notes": "Demo pipeline: resized, normalized, mock-classified, simulated Grad-CAM."
}
```

### Frontend state
- `uploadedImage`
- `predictionResult`
- `clinicianDecision`
- `clinicianComment`
- `loading`
- `error`

## Build Order
1. Scaffold `Next.js` frontend and `FastAPI` backend.
2. Implement backend `/health` and `/predict` with hardcoded demo output first.
3. Add real image preprocessing in backend.
4. Add simulated label/confidence logic.
5. Add simulated heatmap generation.
6. Add template-based explanation generator.
7. Build frontend upload and result display flow.
8. Add clinician accept/reject/comment controls.
9. Add disclaimer, loading, and error states.
10. Run one full demo walkthrough and polish labels/layout.

## Test Plan
- Upload a valid image and confirm full result rendering.
- Upload an unsupported file and confirm friendly error handling.
- Confirm original image preview appears before inference completes.
- Confirm dashboard displays:
  - predicted label
  - confidence
  - heatmap
  - explanation
- Confirm clinician can:
  - accept result
  - reject result
  - enter comment
- Confirm backend `/health` responds successfully.
- Confirm app still works if classifier stays mocked and no real model file exists.

## Assumptions And Defaults
- Stack is `Next.js + FastAPI`.
- This is demo-only, so no authentication, database, audit trail, or real clinical validation.
- Prediction and Grad-CAM can be simulated as long as the user experience matches the intended workflow.
- Heatmap may be a generated visual overlay rather than a true model-derived Grad-CAM.
- Only one-image-at-a-time flow is needed.
- Local file/temp storage is acceptable for the demo.
- The dashboard is single-user and session-only.
