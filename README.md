# DrCancer Demo

Breast ultrasound review workflow using a `Next.js` frontend and `FastAPI` backend. The codebase now supports:

- TensorFlow transfer-learning inference when a trained model is present
- Grad-CAM heatmap generation for real model explanations
- A fallback heuristic mode so the demo still works before the TensorFlow model is trained
- AI draft reporting plus clinician review in the frontend

## Frontend

```bash
npm install
npm run dev
```

The UI expects the backend at `http://127.0.0.1:8000` by default. Override with:

```bash
$env:NEXT_PUBLIC_API_BASE_URL="http://127.0.0.1:8000"
```

## Backend

```bash
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

The backend looks for a trained model at:

```text
backend/models/breast_ultrasound_classifier.keras
```

If the model is missing, the app stays usable in fallback mode and the health endpoint reports that status.

## Train The TensorFlow Model

Prepare your dataset with this structure:

```text
your-dataset/
  benign/
  malignant/
  normal/
```

Then run:

```bash
python backend/train.py --data-dir "C:\path\to\your-dataset"
```

Optional knobs:

```bash
python backend/train.py --data-dir "C:\path\to\your-dataset" --epochs-head 8 --epochs-finetune 5 --batch-size 16
```

Training outputs:

- `backend/models/breast_ultrasound_classifier.keras`
- `backend/models/breast_ultrasound_classifier.metadata.json`

After training, restart the FastAPI server so it loads the saved model.

## Demo Notes

- Upload one image at a time.
- Results are session-only and not stored in a database.
- If TensorFlow or the trained model is unavailable, the backend falls back to heuristic classification with a simulated heatmap.
- When the trained model is present, predictions use TensorFlow transfer learning and Grad-CAM.
- Not for clinical use.
