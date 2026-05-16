from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFilter

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - tensorflow is optional until installed
    tf = None


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "breast_ultrasound_classifier.keras"
METADATA_PATH = MODEL_DIR / "breast_ultrasound_classifier.metadata.json"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
DEFAULT_CLASS_NAMES = ["benign", "malignant", "normal"]
IMAGE_SIZE = (224, 224)


@dataclass
class InferenceResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    heatmap_path: Path
    explanation: str
    report_draft: str
    processing_notes: str
    xai_method: str
    model_mode: str


class InferenceService:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.class_names = DEFAULT_CLASS_NAMES.copy()
        self.last_conv_layer_name: str | None = None
        self.backbone_layer_name: str | None = None
        self.model_mode = "heuristic-fallback"
        self.status_message = "TensorFlow model not loaded."
        self._load_model()

    def _load_model(self) -> None:
        if tf is None:
            self.status_message = (
                "TensorFlow is not installed. Using heuristic fallback until dependencies are installed."
            )
            return

        if not MODEL_PATH.exists():
            self.status_message = (
                f"Model file not found at {MODEL_PATH.name}. Train the classifier to enable real inference."
            )
            return

        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            self.class_names = self._load_class_names()
            self.backbone_layer_name, self.last_conv_layer_name = self._find_backbone_details()
            self.model_mode = "tensorflow-transfer-learning"
            self.status_message = f"Loaded TensorFlow model from {MODEL_PATH.name}."
        except Exception as exc:  # pragma: no cover - defensive guard for environment/model mismatch
            self.model = None
            self.class_names = DEFAULT_CLASS_NAMES.copy()
            self.backbone_layer_name = None
            self.last_conv_layer_name = None
            self.model_mode = "heuristic-fallback"
            self.status_message = f"Failed to load TensorFlow model: {exc}"

    def _load_class_names(self) -> list[str]:
        if not METADATA_PATH.exists():
            return DEFAULT_CLASS_NAMES.copy()

        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            class_names = metadata.get("class_names")
            if isinstance(class_names, list) and all(
                isinstance(class_name, str) for class_name in class_names
            ):
                return class_names
        except json.JSONDecodeError:
            pass

        return DEFAULT_CLASS_NAMES.copy()

    def _find_backbone_details(self) -> tuple[str | None, str | None]:
        if self.model is None:
            return None, None

        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.Model):
                for nested_layer in reversed(layer.layers):
                    output_shape = getattr(nested_layer, "output_shape", None)
                    if output_shape is not None and len(output_shape) == 4:
                        return layer.name, nested_layer.name

            output_shape = getattr(layer, "output_shape", None)
            if output_shape is not None and len(output_shape) == 4:
                return layer.name, layer.name

        return None, None

    def predict(self, image: Image.Image) -> InferenceResult:
        processed, normalized = preprocess_image(image)

        if (
            self.model is not None
            and tf is not None
            and self.last_conv_layer_name
            and self.backbone_layer_name
        ):
            probabilities = self._predict_probabilities(normalized)
            label = max(probabilities, key=probabilities.get)
            confidence = probabilities[label]
            heatmap = self._generate_gradcam(processed, normalized, label)
            explanation = generate_explanation(label, confidence, real_model=True)
            report_draft = generate_report_draft(label, confidence, probabilities, real_model=True)
            processing_notes = (
                "TensorFlow transfer-learning pipeline: resized, normalized, EfficientNet inference, Grad-CAM generated."
            )
            xai_method = "Grad-CAM"
            model_mode = self.model_mode
        else:
            label, confidence, focus_region = classify_demo(normalized)
            probabilities = build_fallback_probabilities(label, confidence)
            heatmap = create_demo_heatmap(processed, label, focus_region)
            explanation = generate_explanation(label, confidence, real_model=False)
            report_draft = generate_report_draft(label, confidence, probabilities, real_model=False)
            processing_notes = (
                "Fallback pipeline: resized, normalized, heuristic classification, simulated heatmap. "
                "Train and save the TensorFlow model to enable real inference."
            )
            xai_method = "Simulated focal overlay"
            model_mode = self.model_mode

        return InferenceResult(
            label=label,
            confidence=round(confidence, 4),
            probabilities={key: round(value, 4) for key, value in probabilities.items()},
            heatmap_path=heatmap,
            explanation=explanation,
            report_draft=report_draft,
            processing_notes=processing_notes,
            xai_method=xai_method,
            model_mode=model_mode,
        )

    def _predict_probabilities(self, normalized: np.ndarray) -> dict[str, float]:
        assert self.model is not None
        batch = np.expand_dims(normalized, axis=0)
        predictions = self.model.predict(batch, verbose=0)[0]
        probabilities = tf.nn.softmax(predictions).numpy() if predictions.ndim else predictions

        if predictions.ndim == 1 and np.isclose(np.sum(predictions), 1.0, atol=1e-3):
            probabilities = predictions

        return {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(self.class_names)
        }

    def _generate_gradcam(
        self, image: Image.Image, normalized: np.ndarray, predicted_label: str
    ) -> Path:
        assert self.model is not None
        assert tf is not None
        assert self.last_conv_layer_name is not None
        assert self.backbone_layer_name is not None

        backbone = self.model.get_layer(self.backbone_layer_name)
        classifier_layers = [
            self.model.get_layer("global_avg_pool"),
            self.model.get_layer("dropout"),
            self.model.get_layer("prediction"),
        ]

        input_tensor = tf.convert_to_tensor(np.expand_dims(normalized, axis=0), dtype=tf.float32)
        target_index = self.class_names.index(predicted_label)

        with tf.GradientTape() as tape:
            conv_outputs = backbone(input_tensor, training=False)
            predictions = conv_outputs
            for layer in classifier_layers:
                predictions = layer(predictions, training=False)
            target_score = predictions[:, target_index]

        gradients = tape.gradient(target_score, conv_outputs)
        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.reduce_max(heatmap)

        if float(max_value) == 0.0:
            return create_demo_heatmap(image, predicted_label, (0.5, 0.5))

        heatmap = heatmap / max_value
        heatmap_array = heatmap.numpy()
        return create_gradcam_overlay(image, heatmap_array, predicted_label)


inference_service = InferenceService()

app = FastAPI(title="DrCancer Demo Backend", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model_mode": inference_service.model_mode,
        "model_status": inference_service.status_message,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, object]:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload PNG, JPG, JPEG, or WEBP.",
        )

    raw_bytes = await image.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        original = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive guard for corrupted input
        raise HTTPException(status_code=400, detail="The image could not be opened.") from exc

    result = inference_service.predict(original)

    return {
        "label": result.label,
        "confidence": round(result.confidence, 2),
        "probabilities": result.probabilities,
        "heatmap_url": f"/outputs/{result.heatmap_path.name}",
        "explanation": result.explanation,
        "report_draft": result.report_draft,
        "processing_notes": result.processing_notes,
        "xai_method": result.xai_method,
        "model_mode": result.model_mode,
    }


def preprocess_image(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    resized = image.resize(IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32)
    return resized, array


def classify_demo(array: np.ndarray) -> tuple[str, float, tuple[float, float]]:
    array = array / 255.0
    grayscale = array.mean(axis=2)
    brightness = float(grayscale.mean())
    contrast = float(grayscale.std())

    malignant_score = np.clip((0.52 - brightness) * 1.8 + contrast * 2.2, 0.0, 1.0)
    benign_score = np.clip(0.55 - abs(brightness - 0.5) + contrast * 1.1, 0.0, 1.0)
    normal_score = np.clip(brightness * 1.2 - contrast * 0.9, 0.0, 1.0)

    scores = {
        "normal": float(normal_score),
        "benign": float(benign_score),
        "malignant": float(malignant_score),
    }

    label = max(scores, key=scores.get)
    confidence = float(0.65 + scores[label] * 0.3)
    focus_x = float(np.clip(0.2 + contrast * 2.0, 0.2, 0.8))
    focus_y = float(np.clip(0.8 - brightness, 0.2, 0.8))
    return label, confidence, (focus_x, focus_y)


def build_fallback_probabilities(label: str, confidence: float) -> dict[str, float]:
    remainder = max(0.0, 1.0 - confidence)
    other_labels = [item for item in DEFAULT_CLASS_NAMES if item != label]
    shared = remainder / len(other_labels)
    scores = {other_label: shared for other_label in other_labels}
    scores[label] = confidence
    return scores


def create_gradcam_overlay(image: Image.Image, heatmap: np.ndarray, label: str) -> Path:
    heatmap_image = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255), mode="L")
    heatmap_image = heatmap_image.resize(image.size).filter(ImageFilter.GaussianBlur(radius=6))
    heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0

    color = np.array(
        [255, 96, 96] if label == "malignant" else [255, 190, 84] if label == "benign" else [96, 178, 255],
        dtype=np.float32,
    )
    base = np.asarray(image, dtype=np.float32)
    overlay_strength = np.expand_dims(heatmap_array * 0.75, axis=-1)
    overlay = base * (1.0 - overlay_strength) + color * overlay_strength
    combined = Image.fromarray(np.uint8(np.clip(overlay, 0, 255)))

    filename = f"heatmap-{uuid.uuid4().hex}.png"
    output_path = OUTPUT_DIR / filename
    combined.save(output_path, format="PNG")
    return output_path


def create_demo_heatmap(
    image: Image.Image, label: str, focus_region: tuple[float, float]
) -> Path:
    width, height = image.size
    center_x = width * focus_region[0]
    center_y = height * focus_region[1]
    radius = min(width, height) * (
        0.16 if label == "normal" else 0.24 if label == "benign" else 0.3
    )

    base = image.copy()
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for ring in range(6, 0, -1):
        current_radius = radius * ring / 6
        alpha = int(28 + ring * 16)
        color = (
            (255, 92, 92, alpha)
            if label == "malignant"
            else (255, 188, 79, alpha)
            if label == "benign"
            else (96, 178, 255, alpha)
        )
        bbox = [
            center_x - current_radius,
            center_y - current_radius,
            center_x + current_radius,
            center_y + current_radius,
        ]
        draw.ellipse(bbox, fill=color)

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=18))
    combined = Image.alpha_composite(base.convert("RGBA"), overlay)

    guide = ImageDraw.Draw(combined)
    guide.rounded_rectangle(
        [
            max(center_x - radius * 0.95, 12),
            max(center_y - radius * 0.95, 12),
            min(center_x + radius * 0.95, width - 12),
            min(center_y + radius * 0.95, height - 12),
        ],
        radius=18,
        outline=(255, 250, 240, 180),
        width=3,
    )

    filename = f"heatmap-{uuid.uuid4().hex}.png"
    output_path = OUTPUT_DIR / filename
    combined.convert("RGB").save(output_path, format="PNG")
    return output_path


def generate_explanation(label: str, confidence: float, real_model: bool) -> str:
    confidence_band = "high" if confidence >= 0.85 else "medium" if confidence >= 0.65 else "low"
    model_prefix = "The TensorFlow classifier" if real_model else "The fallback demo classifier"

    templates = {
        ("normal", "high"): f"{model_prefix} does not emphasize clearly suspicious features, which is more consistent with a normal appearance.",
        ("normal", "medium"): f"{model_prefix} leans toward a normal appearance, though the scan should still be reviewed with clinical context.",
        ("normal", "low"): f"{model_prefix} does not strongly indicate suspicious features, but the separation between classes is limited.",
        ("benign", "high"): f"{model_prefix} highlights a less suspicious pattern overall, which is more consistent with a benign-style finding.",
        ("benign", "medium"): f"{model_prefix} suggests a benign-leaning pattern, though clinician review remains important before concluding the case.",
        ("benign", "low"): f"{model_prefix} mildly favors a benign pattern, but the prediction margin is modest.",
        ("malignant", "high"): f"{model_prefix} highlights suspicious structure in the emphasized region, so further diagnostic review is advised.",
        ("malignant", "medium"): f"{model_prefix} marks the highlighted region as more suspicious and worth closer clinician attention.",
        ("malignant", "low"): f"{model_prefix} shows some suspicious emphasis, although confidence remains limited.",
    }
    return templates[(label, confidence_band)]


def generate_report_draft(
    label: str, confidence: float, probabilities: dict[str, float], real_model: bool
) -> str:
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    differential = ", ".join(f"{class_name}: {score * 100:.1f}%" for class_name, score in ranked)
    system_name = "TensorFlow transfer-learning model" if real_model else "fallback heuristic pipeline"

    recommendation = {
        "normal": "Correlate with the full ultrasound exam and routine follow-up guidance.",
        "benign": "Correlate with morphology, margins, and standard benign workup guidance.",
        "malignant": "Recommend closer radiology review and diagnostic escalation as clinically appropriate.",
    }[label]

    return (
        f"AI draft report: The {system_name} classified this ultrasound as {label} "
        f"with {confidence * 100:.1f}% confidence. Class probabilities were {differential}. "
        f"Highlighted regions reflect the model attention map used for visual explanation. "
        f"{recommendation} Demo output only and not for clinical use."
    )
