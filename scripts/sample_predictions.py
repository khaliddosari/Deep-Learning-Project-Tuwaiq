"""Real head-to-head predictions for the ten demo sample images.

The web demo shows, for one CIFAR-10 image per class, what the shipped Dense
model predicts against what the fine-tuned MobileNetV2 predicts. Those numbers
have to come from the models themselves, so this script produces them:

1. Rebuilds the Part 15 final model exactly as the notebook builds it
   (Flatten -> 3 x [Dense 512 relu + Dropout 0.3] -> Dense 10 softmax, Adam at
   1e-4, batch 128, 50 epochs, EarlyStopping on val_loss with patience 5 and
   restore_best_weights) and trains it on the committed split.
2. Evaluates it on the sealed test set, so the demo can state the accuracy of
   the exact weights the sample predictions come from.
3. Loads the fine-tuned MobileNetV2 from models/.
4. Runs both models over the ten test indices named in
   Demo/public/samples/samples.json and records the full probability vector.

Writes Results/sample_predictions.json and saves the Dense weights to
models/dense_final.keras so the Gradio app can serve a trained model instead of
building a fresh untrained one.

Run from the project root:  python scripts/sample_predictions.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_NAMES, SEED  # noqa: E402
from src.data import load_splits  # noqa: E402
from src.utils import set_seed  # noqa: E402

SAMPLES_JSON = PROJECT_ROOT / "Demo" / "public" / "samples" / "samples.json"
MOBILENET_PATH = PROJECT_ROOT / "models" / "mobilenet_cifar10.keras"
DENSE_PATH = PROJECT_ROOT / "models" / "dense_final.keras"
OUT_PATH = PROJECT_ROOT / "Results" / "sample_predictions.json"

EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
DROPOUT = 0.3
HIDDEN_UNITS = [512, 512, 512]


def build_final_model():
    """The Part 15 final model, layer for layer as the notebook builds it."""
    from tensorflow import keras
    from tensorflow.keras import layers

    set_seed(SEED)

    model = keras.Sequential(
        [keras.Input(shape=(32, 32, 3)), layers.Flatten()]
        + [
            layer
            for units in HIDDEN_UNITS
            for layer in (layers.Dense(units, activation="relu"), layers.Dropout(DROPOUT))
        ]
        + [layers.Dense(10, activation="softmax")]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    from tensorflow import keras
    from sklearn.metrics import f1_score, precision_score, recall_score

    splits = load_splits()
    samples = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))

    # The demo's PNGs were exported from the test set. Confirm each one still
    # matches the index it claims, so a prediction is never attached to the
    # wrong picture.
    for s in samples:
        idx = s["test_index"]
        assert int(splits.y_test[idx]) == s["class_id"], (
            f"{s['filename']}: test index {idx} is class "
            f"{int(splits.y_test[idx])}, not {s['class_id']}"
        )

    print(f"training the Part 15 final model for up to {EPOCHS} epochs ...")
    dense = build_final_model()
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    t0 = time.time()
    history = dense.fit(
        splits.X_train,
        splits.y_train,
        validation_data=(splits.X_val, splits.y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        verbose=2,
    ).history
    train_seconds = time.time() - t0

    epochs_run = len(history["loss"])
    best_val_epoch = int(np.argmax(history["val_accuracy"])) + 1
    dense_loss, dense_accuracy = dense.evaluate(splits.X_test, splits.y_test, verbose=0)

    dense_probs_all = dense.predict(splits.X_test, verbose=0)
    dense_pred_all = dense_probs_all.argmax(axis=1)
    dense_metrics = {
        "test_accuracy": round(float(dense_accuracy), 4),
        "test_loss": round(float(dense_loss), 4),
        "weighted_precision": round(
            float(precision_score(splits.y_test, dense_pred_all, average="weighted", zero_division=0)), 4
        ),
        "weighted_recall": round(
            float(recall_score(splits.y_test, dense_pred_all, average="weighted", zero_division=0)), 4
        ),
        "weighted_f1": round(
            float(f1_score(splits.y_test, dense_pred_all, average="weighted", zero_division=0)), 4
        ),
        "epochs_run": epochs_run,
        "stopped_early": epochs_run < EPOCHS,
        "best_val_accuracy": round(float(max(history["val_accuracy"])), 4),
        "best_val_accuracy_epoch": best_val_epoch,
        "train_seconds": round(train_seconds, 1),
    }
    print("dense:", dense_metrics)

    DENSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    dense.save(DENSE_PATH)
    print(f"saved {DENSE_PATH}")

    print("loading MobileNetV2 ...")
    mobilenet = keras.models.load_model(MOBILENET_PATH)

    idx = np.array([s["test_index"] for s in samples])
    batch = splits.X_test[idx]
    dense_probs = dense.predict(batch, verbose=0)
    cnn_probs = mobilenet.predict(batch, verbose=0)

    rows = []
    for i, s in enumerate(samples):
        d, c = dense_probs[i], cnn_probs[i]
        di, ci = int(d.argmax()), int(c.argmax())
        rows.append(
            {
                "name": s["class_name"],
                "class_id": s["class_id"],
                "test_index": s["test_index"],
                "src": s["src"],
                "dense_pred": CLASS_NAMES[di],
                "dense_confidence": round(float(d[di]) * 100, 1),
                "dense_true_class_confidence": round(float(d[s["class_id"]]) * 100, 1),
                "dense_correct": di == s["class_id"],
                "cnn_pred": CLASS_NAMES[ci],
                "cnn_confidence": round(float(c[ci]) * 100, 1),
                "cnn_true_class_confidence": round(float(c[s["class_id"]]) * 100, 1),
                "cnn_correct": ci == s["class_id"],
            }
        )
        print(
            f"  {s['class_name']:<11} dense={rows[-1]['dense_pred']:<11}"
            f"{rows[-1]['dense_confidence']:>5.1f}%   "
            f"cnn={rows[-1]['cnn_pred']:<11}{rows[-1]['cnn_confidence']:>5.1f}%"
        )

    payload = {
        "note": (
            "Generated by scripts/sample_predictions.py. The Dense model is the "
            "Part 15 final model rebuilt and retrained from the same recipe and "
            "seed, so its test accuracy differs slightly from the notebook run "
            "for the reasons Results/part15_rerun.json documents."
        ),
        "seed": SEED,
        "dense": dense_metrics,
        "samples": rows,
        "dense_correct_count": sum(r["dense_correct"] for r in rows),
        "cnn_correct_count": sum(r["cnn_correct"] for r in rows),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    print(f"dense correct {payload['dense_correct_count']}/10, cnn correct {payload['cnn_correct_count']}/10")


if __name__ == "__main__":
    main()
