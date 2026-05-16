from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CACHE_DIR = MODEL_DIR / ".keras-cache"
MODEL_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("KERAS_HOME", str(CACHE_DIR))

import numpy as np
import tensorflow as tf


IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["benign", "malignant", "normal"]
MODEL_PATH = MODEL_DIR / "breast_ultrasound_classifier.keras"
METADATA_PATH = MODEL_DIR / "breast_ultrasound_classifier.metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the breast ultrasound TensorFlow classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing benign/, malignant/, and normal/ subfolders.",
    )
    parser.add_argument("--epochs-head", type=int, default=6, help="Epochs for frozen-base training.")
    parser.add_argument("--epochs-finetune", type=int, default=4, help="Epochs for fine-tuning.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument(
        "--fine-tune-learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate used after unfreezing the top backbone layers.",
    )
    parser.add_argument(
        "--balance-strategy",
        choices=["oversample", "none"],
        default="oversample",
        help="How to rebalance the training split before fitting.",
    )
    parser.add_argument(
        "--loss",
        choices=["focal", "crossentropy"],
        default="focal",
        help="Training loss. Focal loss helps emphasize harder minority-class errors.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma parameter used when focal loss is enabled.",
    )
    return parser.parse_args()


def collect_examples(data_dir: Path) -> tuple[list[str], list[int]]:
    filepaths: list[str] = []
    labels: list[int] = []

    for index, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            if "_mask" in image_path.stem.lower():
                continue
            filepaths.append(str(image_path))
            labels.append(index)

    if not filepaths:
        raise ValueError("No supported image files were found in the dataset directory.")

    return filepaths, labels


def stratified_split(
    filepaths: list[str], labels: list[int], seed: int
) -> tuple[list[str], list[int], list[str], list[int], list[str], list[int]]:
    rng = random.Random(seed)
    grouped: dict[int, list[str]] = {index: [] for index in range(len(CLASS_NAMES))}

    for path, label in zip(filepaths, labels, strict=True):
        grouped[label].append(path)

    train_paths: list[str] = []
    train_labels: list[int] = []
    val_paths: list[str] = []
    val_labels: list[int] = []
    test_paths: list[str] = []
    test_labels: list[int] = []

    for label, paths in grouped.items():
        rng.shuffle(paths)
        total = len(paths)
        test_count = max(1, math.floor(total * 0.15))
        val_count = max(1, math.floor(total * 0.15))

        test_subset = paths[:test_count]
        val_subset = paths[test_count : test_count + val_count]
        train_subset = paths[test_count + val_count :]

        train_paths.extend(train_subset)
        train_labels.extend([label] * len(train_subset))
        val_paths.extend(val_subset)
        val_labels.extend([label] * len(val_subset))
        test_paths.extend(test_subset)
        test_labels.extend([label] * len(test_subset))

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def decode_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, depth=len(CLASS_NAMES))
    return image, label


def build_dataset(
    filepaths: list[str], labels: list[int], batch_size: int, training: bool
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        dataset = dataset.shuffle(len(filepaths), seed=42, reshuffle_each_iteration=True)

    dataset = dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
            ],
            name="augmentation",
        )

        dataset = dataset.map(
            lambda image, label: (augmentation(image, training=True), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="prediction")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="breast_ultrasound_classifier")

    return model, base_model


def compute_class_weights(labels: list[int]) -> dict[int, float]:
    counts = Counter(labels)
    total = sum(counts.values())
    num_classes = len(CLASS_NAMES)
    return {
        label: total / (num_classes * class_count)
        for label, class_count in counts.items()
    }


def rebalance_training_split(
    filepaths: list[str], labels: list[int], strategy: str, seed: int
) -> tuple[list[str], list[int], dict[str, int]]:
    original_counts = Counter(labels)
    summary = {
        class_name: original_counts.get(index, 0) for index, class_name in enumerate(CLASS_NAMES)
    }

    if strategy == "none":
        return filepaths, labels, summary

    if strategy != "oversample":
        raise ValueError(f"Unsupported balance strategy: {strategy}")

    rng = random.Random(seed)
    grouped: dict[int, list[str]] = {index: [] for index in range(len(CLASS_NAMES))}
    for path, label in zip(filepaths, labels, strict=True):
        grouped[label].append(path)

    target_count = max(len(paths) for paths in grouped.values())
    balanced_paths: list[str] = []
    balanced_labels: list[int] = []

    for label, paths in grouped.items():
        expanded_paths = paths.copy()
        if len(expanded_paths) < target_count:
            expanded_paths.extend(rng.choices(paths, k=target_count - len(expanded_paths)))
        rng.shuffle(expanded_paths)
        balanced_paths.extend(expanded_paths)
        balanced_labels.extend([label] * len(expanded_paths))

    combined = list(zip(balanced_paths, balanced_labels, strict=True))
    rng.shuffle(combined)
    balanced_paths = [path for path, _ in combined]
    balanced_labels = [label for _, label in combined]
    summary = {class_name: target_count for class_name in CLASS_NAMES}
    return balanced_paths, balanced_labels, summary


def build_loss(loss_name: str, gamma: float) -> tf.keras.losses.Loss | str:
    if loss_name == "crossentropy":
        return "categorical_crossentropy"
    if loss_name == "focal":
        return tf.keras.losses.CategoricalFocalCrossentropy(gamma=gamma)
    raise ValueError(f"Unsupported loss: {loss_name}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    filepaths, labels = collect_examples(args.data_dir)
    (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
    ) = stratified_split(filepaths, labels, args.seed)

    original_train_distribution = {
        class_name: train_labels.count(index) for index, class_name in enumerate(CLASS_NAMES)
    }
    train_paths, train_labels, balanced_train_distribution = rebalance_training_split(
        train_paths,
        train_labels,
        args.balance_strategy,
        args.seed,
    )

    train_dataset = build_dataset(train_paths, train_labels, args.batch_size, training=True)
    val_dataset = build_dataset(val_paths, val_labels, args.batch_size, training=False)
    test_dataset = build_dataset(test_paths, test_labels, args.batch_size, training=False)

    model, base_model = build_model()
    class_weights = compute_class_weights(train_labels)
    loss = build_loss(args.loss, args.focal_gamma)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs_head,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.fine_tune_learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs_finetune,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    model.save(MODEL_PATH)

    metadata = {
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "splits": {
            "train": len(train_paths),
            "validation": len(val_paths),
            "test": len(test_paths),
        },
        "class_distribution": {
            class_name: labels.count(index) for index, class_name in enumerate(CLASS_NAMES)
        },
        "test_metrics": {
            "loss": float(test_loss),
            "accuracy": float(test_accuracy),
        },
        "training_configuration": {
            "balance_strategy": args.balance_strategy,
            "loss": args.loss,
            "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
            "original_train_distribution": original_train_distribution,
            "balanced_train_distribution": balanced_train_distribution,
            "class_weights": {
                CLASS_NAMES[label]: round(weight, 4) for label, weight in class_weights.items()
            },
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
