"""Part 15, supporting experiment: does L2 add anything on top of Dropout?

Why this exists
---------------
Part 15 now ships Dropout 0.3 + Early Stopping, on the evidence of Part 12 (which
ranked Dropout first of four regularizers) and Part 14 (which ablates Dropout and
finds a decisive effect). Two questions that leaves open:

  1. Should L2 have been kept as well? Part 14 tests Dropout on/off with no L2 in
     either arm, so it cannot answer this.
  2. Part 15's 50-epoch budget binds -- Early Stopping never fired and validation
     loss was still falling at epoch 49 -- so its score is a floor, not a
     converged result. What happens with room to converge?

This 2x2 answers both: Dropout on/off crossed with L2 on/off, 100 epochs so
Early Stopping can actually fire, everything else identical.

Protocol
--------
Identical in every respect except the regularizer:

    architecture   [512, 512, 512] Dense, ReLU, softmax output
    optimizer      Adam, learning rate 0.0001      (Part 9, Part 11)
    batch size     128                             (Part 10)
    epochs         100 maximum (Part 15 used 50, and the cap bound)
    callback       EarlyStopping(val_loss, patience=5, restore_best_weights)
    seed           42, reseeded before every build so all four start from
                   identical initial weights

    A  l2          L2 1e-4 on every hidden kernel
    B  dropout     Dropout 0.3 after every hidden block  <- the final model
    C  both        L2 1e-4 and Dropout 0.3
    D  neither     Early Stopping alone (control)

Model construction matches the notebook's Part 12 and Part 15 cells exactly:
Dropout sits after each hidden activation, L2 applies to every hidden kernel.

Selection is made on best validation accuracy, which is the quantity
restore_best_weights recovers. The test set is touched only after that decision
is recorded, and no tuning follows it -- the four test scores are reported for
transparency, not used to choose.

Run
---
    LD_LIBRARY_PATH=$(find .venv/lib/python3.12/site-packages/nvidia -name lib -type d | tr '\\n' ':') \\
        .venv/bin/python scripts/part15_rerun.py

Writes Results/part15_rerun.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.config import IMG_SHAPE, NUM_CLASSES, RESULTS_DIR, SEED
from src.data import load_splits
from src.models import validate_model
from src.utils import set_seed

LEARNING_RATE = 0.0001
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 5
HIDDEN_UNITS = [512, 512, 512]
DROPOUT_RATE = 0.3
L2_STRENGTH = 0.0001

# (key, label, uses l2, uses dropout)
CONFIGS = [
    ("l2", "L2 only", True, False),
    ("dropout", "Dropout only (final model)", False, True),
    ("both", "Dropout + L2", True, True),
    ("neither", "Early Stopping only (control)", False, False),
]


def build(use_l2: bool, use_dropout: bool):
    """Build one variant, reseeded so every config starts from the same weights."""
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    set_seed(SEED)

    reg = regularizers.l2(L2_STRENGTH) if use_l2 else None

    net: list = [keras.Input(shape=IMG_SHAPE), layers.Flatten()]
    for units in HIDDEN_UNITS:
        net.append(layers.Dense(units, activation="relu", kernel_regularizer=reg))
        if use_dropout:
            net.append(layers.Dropout(DROPOUT_RATE))
    net.append(layers.Dense(NUM_CLASSES, activation="softmax"))

    model = keras.Sequential(net)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    validate_model(model)  # no forbidden layers, 10-unit output
    return model


def main() -> None:
    from tensorflow import keras

    data = load_splits()
    print(data.summary(), flush=True)

    results = {}

    for key, label, use_l2, use_dropout in CONFIGS:
        print("\n" + "=" * 66, flush=True)
        print(f"{label}", flush=True)
        print("=" * 66, flush=True)

        model = build(use_l2, use_dropout)

        stopper = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
        )

        history = model.fit(
            data.X_train,
            data.y_train,
            validation_data=(data.X_val, data.y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[stopper],
            verbose=2,
        ).history

        val_acc = history["val_accuracy"]
        val_loss = history["val_loss"]
        best_epoch = int(np.argmax(val_acc)) + 1
        epochs_run = len(val_acc)

        # restore_best_weights has already put the best-val_loss weights back,
        # so this is the model that would actually ship.
        test_loss, test_accuracy = model.evaluate(data.X_test, data.y_test, verbose=0)

        results[key] = {
            "label": label,
            "l2": L2_STRENGTH if use_l2 else None,
            "dropout": DROPOUT_RATE if use_dropout else None,
            "epochs_run": epochs_run,
            "stopped_early": epochs_run < EPOCHS,
            "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
            "best_val_accuracy": float(max(val_acc)),
            "best_val_accuracy_epoch": best_epoch,
            "min_val_loss": float(min(val_loss)),
            "min_val_loss_epoch": int(np.argmin(val_loss)) + 1,
            "final_train_accuracy": float(history["accuracy"][-1]),
            "final_val_accuracy": float(val_acc[-1]),
            # GAP_PEAK, the definition Parts 4-7 and 19 use.
            "gap_peak": float(history["accuracy"][-1] - max(val_acc)),
            # GAP_FINAL, the definition Parts 12-14 use.
            "gap_final": float(history["accuracy"][-1] - val_acc[-1]),
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "history": {k: [float(v) for v in vals] for k, vals in history.items()},
        }

        r = results[key]
        print(
            f"\n  epochs run          {epochs_run}"
            f"{' (early stopped)' if r['stopped_early'] else f' (hit the {EPOCHS}-epoch cap)'}"
            f"\n  best val accuracy   {r['best_val_accuracy']:.4f} (epoch {best_epoch})"
            f"\n  min val loss        {r['min_val_loss']:.4f}"
            f"\n  final train acc     {r['final_train_accuracy']:.4f}"
            f"\n  gap (peak)          {r['gap_peak']:.4f}"
            f"\n  test accuracy       {r['test_accuracy']:.4f}",
            flush=True,
        )

    winner = max(results, key=lambda k: results[k]["best_val_accuracy"])
    incumbent = "dropout"

    summary = {
        "protocol": {
            "architecture": HIDDEN_UNITS,
            "optimizer": "adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": EPOCHS,
            "early_stopping": {"monitor": "val_loss", "patience": PATIENCE,
                               "restore_best_weights": True},
            "seed": SEED,
            "selected_on": "best validation accuracy",
        },
        "runs": results,
        "winner": winner,
        "incumbent": incumbent,
        "delta_points": round(
            (results[winner]["best_val_accuracy"] - results[incumbent]["best_val_accuracy"]) * 100, 2
        ),
        "delta_images": round(
            (results[winner]["best_val_accuracy"] - results[incumbent]["best_val_accuracy"]) * len(data.y_val)
        ),
        "validation_size": int(len(data.y_val)),
    }

    out = RESULTS_DIR / "part15_rerun.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 66, flush=True)
    print("RESULT", flush=True)
    print("=" * 66, flush=True)
    for key, _, _, _ in CONFIGS:
        r = results[key]
        mark = "  <-- winner" if key == winner else ""
        print(
            f"  {r['label']:<34} best val {r['best_val_accuracy']:.4f}"
            f"  gap {r['gap_peak']:+.4f}  test {r['test_accuracy']:.4f}{mark}",
            flush=True,
        )
    print(
        f"\n  Winner beats the shipped model by {summary['delta_points']:+.2f} points "
        f"({summary['delta_images']:+d} images out of {summary['validation_size']:,}).",
        flush=True,
    )
    print(f"\n  Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
