"""
train_model.py
Loads every recorded sequence from dataset/, builds the label-encoded
train/validation/test splits, trains the LSTM gesture classifier described
in the project spec, and writes out:
  - saved_models/sign_lstm_model.h5
  - saved_models/label_encoder.pkl
  - saved_models/training_history.json
  - saved_models/accuracy_plot.png
  - saved_models/loss_plot.png
  - saved_models/confusion_matrix.png
  - saved_models/classification_report.txt

Usage:
    python train_model.py
    python train_model.py --epochs 300 --batch-size 32
"""

import os
import json
import pickle
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Masking
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import config
from utils.helpers import normalize_sequence_length


def load_dataset():
    """Walk dataset/<gesture>/<user>/*.npy and build X, y arrays."""
    X, y = [], []
    gestures_found = sorted(os.listdir(config.DATASET_DIR)) if os.path.isdir(config.DATASET_DIR) else []
    gestures_found = [g for g in gestures_found if os.path.isdir(os.path.join(config.DATASET_DIR, g))]

    if not gestures_found:
        raise RuntimeError(
            "No dataset found. Run collect_dataset.py first to record gesture "
            "sequences before training."
        )

    for gesture in gestures_found:
        gesture_path = os.path.join(config.DATASET_DIR, gesture)
        for user in os.listdir(gesture_path):
            user_path = os.path.join(gesture_path, user)
            if not os.path.isdir(user_path):
                continue
            for fname in os.listdir(user_path):
                if fname.endswith(".npy"):
                    seq = np.load(os.path.join(user_path, fname))
                    seq = normalize_sequence_length(seq)
                    X.append(seq)
                    y.append(gesture)

    if len(X) == 0:
        raise RuntimeError("Dataset folders exist but contain no .npy sequences.")

    return np.array(X, dtype=np.float32), np.array(y)


def build_model(num_classes, sequence_length, feature_size):
    """
    Exact architecture requested in the project specification:
      Input -> LSTM64 -> Dropout -> LSTM128 -> Dropout -> LSTM64
             -> Dense64 -> Dense32 -> Softmax
    """
    model = Sequential(name="sign_language_lstm")
    model.add(Input(shape=(sequence_length, feature_size)))
    model.add(Masking(mask_value=0.0))
    model.add(LSTM(64, return_sequences=True, activation="tanh"))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=True, activation="tanh"))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False, activation="tanh"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, acc_path, loss_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="Val Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history.get("val_loss", []), label="Val Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.5), max(5, len(class_names) * 0.5)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train the sign language LSTM model")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    print("Loading dataset...")
    X, y_raw = load_dataset()
    print(f"Loaded {len(X)} sequences across {len(set(y_raw))} gesture classes.")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_categorical, test_size=(config.TEST_SIZE + config.VAL_SIZE),
        random_state=config.RANDOM_STATE, stratify=y_encoded
    )
    relative_test_size = config.TEST_SIZE / (config.TEST_SIZE + config.VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=config.RANDOM_STATE
    )

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    class_weights_arr = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_encoded), y=y_encoded
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights_arr)}

    model = build_model(num_classes, X.shape[1], X.shape[2])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=config.PATIENCE, restore_best_weights=True),
        ModelCheckpoint(config.MODEL_FILE, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # Ensure the best model is saved even if ModelCheckpoint's monitored
    # metric never improved on a run with very little data
    model.save(config.MODEL_FILE)

    with open(config.LABEL_ENCODER_FILE, "wb") as f:
        pickle.dump(encoder, f)

    with open(config.TRAINING_HISTORY_FILE, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    plot_history(history, config.ACCURACY_PLOT_FILE, config.LOSS_PLOT_FILE)

    print("\nEvaluating on held-out test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}  Test Accuracy: {test_acc:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, encoder.classes_, config.CONFUSION_MATRIX_FILE)

    report = classification_report(y_true, y_pred, target_names=encoder.classes_, zero_division=0)
    with open(config.CLASSIFICATION_REPORT_FILE, "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\nTest Loss: {test_loss:.4f}\n\n")
        f.write(report)
    print(report)

    print(f"\nModel saved to: {config.MODEL_FILE}")
    print(f"Label encoder saved to: {config.LABEL_ENCODER_FILE}")
    print(f"Plots + report saved to: {config.SAVED_MODELS_DIR}")


if __name__ == "__main__":
    main()
