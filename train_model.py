"""
DetectAI - Deep Convolutional Neural Network Training Pipeline
Transfer Learning with EfficientNetB0 / Xception for Face Deepfake Detection.

Features:
- Two-phase training: Feature extraction (frozen base) -> Fine-tuning (unfrozen top layers)
- Data augmentation (rotation, zoom, flips, contrast shifts)
- Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
- Automatic export of training history curves, confusion matrix, and classification report
- Output saved directly to model/model.h5
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Project configuration
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH = MODEL_DIR / "model.h5"
PLOTS_OUTPUT_PATH = MODEL_DIR / "training_curves.png"
CM_OUTPUT_PATH = MODEL_DIR / "confusion_matrix.png"
REPORT_OUTPUT_PATH = MODEL_DIR / "classification_report.txt"

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 15
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5


def create_synthetic_dataset():
    """Generates synthetic dataset for quick verification and model generation."""
    print("[*] Generating synthetic tensor batches for training verification...")
    x = np.random.uniform(0, 255, size=(64, IMG_SIZE[0], IMG_SIZE[1], 3)).astype(np.float32)
    y = np.random.randint(0, 2, size=(64, 1)).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH_SIZE)
    return ds, ds


def create_dataset_generators(dataset_dir: Path, use_synthetic: bool = False):
    """
    Creates tf.data pipelines with train/validation splits and augmentation.
    Expects dataset_dir to contain:
      dataset/
        real/ (class 0: REAL)
        fake/ (class 1: FAKE)
    """
    if use_synthetic:
        return create_synthetic_dataset()

    if not dataset_dir.exists() or not (dataset_dir / "real").exists() or not (dataset_dir / "fake").exists():
        print(f"[!] Dataset directory not found at: {dataset_dir}")
        print("    Please structure your training images as:")
        print(f"      {dataset_dir}/real/*.jpg")
        print(f"      {dataset_dir}/fake/*.jpg")
        print("    (Supports Kaggle '140k Real and Fake Faces' or FaceForensics++)")
        print("    Tip: Run with --synthetic to generate and test model architecture without a dataset.")
        return None, None

    print(f"[*] Loading training dataset from {dataset_dir}...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    # Class labels verification
    print(f"[*] Class names identified: {train_ds.class_names}")

    # Performance optimization: prefetch & cache
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def build_transfer_learning_model():
    """
    Constructs the hybrid CNN architecture:
    EfficientNetB0 backbone (pre-trained on ImageNet) + Custom Classification Head
    """
    # Data Augmentation pipeline integrated directly into the computational graph
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.08),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    # Base transfer learning backbone
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze base model during phase 1
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")
    x = data_augmentation(inputs)
    # EfficientNet expects [0, 255] float inputs as it includes internal Rescaling
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dense(256, activation="relu", name="head_dense_1")(x)
    x = layers.Dropout(0.4, name="head_dropout")(x)
    x = layers.Dense(64, activation="relu", name="head_dense_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="fake_probability")(x)

    model = models.Model(inputs, outputs, name="DetectAI_CNN")
    return model, base_model


def plot_and_save_metrics(history1, history2):
    """
    Generates unified training curves across both training phases and saves to disk.
    """
    acc = history1.history.get("accuracy", [])
    val_acc = history1.history.get("val_accuracy", [])
    loss = history1.history.get("loss", [])
    val_loss = history1.history.get("val_loss", [])

    if history2:
        acc += history2.history.get("accuracy", [])
        val_acc += history2.history.get("val_accuracy", [])
        loss += history2.history.get("loss", [])
        val_loss += history2.history.get("val_loss", [])

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#8b5cf6", linewidth=2)
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#10b981", linewidth=2, linestyle="--")
    if history2:
        plt.axvline(x=len(history1.history["accuracy"]), color="#ef4444", linestyle=":", label="Fine-Tuning Start")
    plt.title("Model Accuracy Evolution", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#8b5cf6", linewidth=2)
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#ef4444", linewidth=2, linestyle="--")
    if history2:
        plt.axvline(x=len(history1.history["loss"]), color="#6d28d9", linestyle=":", label="Fine-Tuning Start")
    plt.title("Model Loss Evolution", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs")
    plt.ylabel("Binary Crossentropy Loss")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_OUTPUT_PATH, dpi=200)
    plt.close()
    print(f"[*] Training curves exported to {PLOTS_OUTPUT_PATH}")


def evaluate_and_generate_reports(model, val_ds):
    """
    Calculates detailed test performance, generates confusion matrix and classification report.
    """
    y_true = []
    y_pred_probs = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().ravel())
        y_pred_probs.extend(preds.ravel())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs >= 0.5).astype(int)

    # Classification report
    target_names = ["REAL", "FAKE"]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(report)

    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("DetectAI - CNN Branch Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
    print(f"[*] Report saved to {REPORT_OUTPUT_PATH}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Purples", ax=ax, values_format="d")
    plt.title("Confusion Matrix - CNN Deepfake Detector", fontweight="bold")
    plt.tight_layout()
    plt.savefig(CM_OUTPUT_PATH, dpi=200)
    plt.close()
    print(f"[*] Confusion matrix saved to {CM_OUTPUT_PATH}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DetectAI CNN Trainer")
    parser.add_argument("--synthetic", action="store_true", help="Train on synthetic batches for quick model compilation")
    parser.add_argument("--epochs1", type=int, default=PHASE1_EPOCHS, help="Phase 1 epochs")
    parser.add_argument("--epochs2", type=int, default=PHASE2_EPOCHS, help="Phase 2 epochs")
    args = parser.parse_args()

    print("==================================================")
    print("   DetectAI - CNN Training Pipeline (EfficientNet)")
    print("==================================================")

    train_ds, val_ds = create_dataset_generators(DATASET_DIR, use_synthetic=args.synthetic)
    if train_ds is None:
        sys.exit(1)

    model, base_model = build_transfer_learning_model()
    model.summary()

    # Training callbacks
    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=str(MODEL_OUTPUT_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    cb_early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    )
    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    )

    # PHASE 1: Train Classification Head Only
    print("\n[*] Starting Phase 1: Training Classification Head...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=INITIAL_LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs1,
        callbacks=[cb_checkpoint, cb_early_stop, cb_reduce_lr],
    )

    # PHASE 2: Unfreeze top layers for fine-tuning
    print("\n[*] Starting Phase 2: Unfreezing Top Layers for Fine-Tuning...")
    base_model.trainable = True
    # Freeze bottom layers, fine-tune the top 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs2,
        callbacks=[cb_checkpoint, cb_early_stop, cb_reduce_lr],
    )

    # Final saving and reporting
    model.save(str(MODEL_OUTPUT_PATH))
    print(f"\n[+] Final trained CNN model saved to: {MODEL_OUTPUT_PATH}")

    plot_and_save_metrics(history_phase1, history_phase2)
    evaluate_and_generate_reports(model, val_ds)
    print("\n[OK] Training complete successfully!")


if __name__ == "__main__":
    main()
