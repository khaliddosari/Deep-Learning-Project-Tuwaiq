"""Train and fine-tune MobileNetV2 on CIFAR-10 with advanced optimizations.

Techniques enabled:
1. In-model data augmentation (flips, translations, rotations, zooms)
2. Bilinear upsampling to 96x96
3. Label smoothing (0.1) on Categorical Cross-Entropy
4. Cosine decay learning rate schedule during fine-tuning
5. Deeper unfreezing (from layer 80 onwards, ~1.9M trainable params)
6. 2 warmup epochs + 4 fine-tuning epochs (Total 6 epochs)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.config import CLASS_NAMES, RESULTS_DIR, SEED
from src.data import load_splits, to_onehot
from src.mobilenet import build_mobilenet_v2, unfreeze_mobilenet_top_layers
from src.utils import set_seed

MODELS_DIR = ROOT / "models"
WARMUP_EPOCHS = 2
FINETUNE_EPOCHS = 4  # Total 6 epochs
BATCH_SIZE = 128


def main():
    print("=" * 75)
    print("  High-Performance MobileNetV2 Fine-Tuning (>91%+ Target on CIFAR-10)")
    print("=" * 75)

    set_seed(SEED)
    data = load_splits()
    print("\nDataset Summary:")
    print(data.summary())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # One-hot encoded labels for label smoothing
    y_train_oh = to_onehot(data.y_train)
    y_val_oh = to_onehot(data.y_val)
    y_test_oh = to_onehot(data.y_test)

    print("\n[1/3] Building augmented MobileNetV2 model (96x96 receptive field)...")
    model, base_model = build_mobilenet_v2(
        target_dim=96,
        head_units=256,
        dropout_rate=0.3,
        learning_rate=1e-3,
        label_smoothing=0.1,
        use_augmentation=True,
        seed=SEED,
        freeze_base=True,
    )
    total_params = model.count_params()
    trainable_params_phase1 = sum(np.prod(w.shape) for w in model.trainable_weights)
    print(f"Total parameters: {total_params:,}")
    print(f"Phase 1 trainable parameters: {trainable_params_phase1:,}")

    # Phase 1: Warmup classification head
    print(f"\n[2/3] Phase 1: Warmup training head for {WARMUP_EPOCHS} epochs (base frozen)...")
    start_time = time.time()
    h1 = model.fit(
        data.X_train,
        y_train_oh,
        validation_data=(data.X_val, y_val_oh),
        epochs=WARMUP_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # Phase 2: Deeper fine-tuning with Cosine Decay
    steps_per_epoch = len(data.X_train) // BATCH_SIZE
    total_ft_steps = steps_per_epoch * FINETUNE_EPOCHS
    print(f"\n[3/3] Phase 2: Fine-tuning layers >= 80 for {FINETUNE_EPOCHS} epochs with Cosine Decay ({total_ft_steps} steps)...")
    
    model = unfreeze_mobilenet_top_layers(
        model,
        base_model,
        fine_tune_at=80,
        total_steps=total_ft_steps,
        initial_learning_rate=2e-4,
        label_smoothing=0.1,
    )
    trainable_params_phase2 = sum(np.prod(w.shape) for w in model.trainable_weights)
    print(f"Phase 2 trainable parameters: {trainable_params_phase2:,}")

    h2 = model.fit(
        data.X_train,
        y_train_oh,
        validation_data=(data.X_val, y_val_oh),
        epochs=WARMUP_EPOCHS + FINETUNE_EPOCHS,
        initial_epoch=WARMUP_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )
    total_train_time = time.time() - start_time

    # Combine training histories
    combined_history = {
        "accuracy": h1.history["accuracy"] + h2.history["accuracy"],
        "loss": h1.history["loss"] + h2.history["loss"],
        "val_accuracy": h1.history["val_accuracy"] + h2.history["val_accuracy"],
        "val_loss": h1.history["val_loss"] + h2.history["val_loss"],
    }

    # Evaluate on 10,000 test images
    print("\nEvaluating on 10,000 sealed test images...")
    eval_start = time.time()
    test_loss, test_acc = model.evaluate(data.X_test, y_test_oh, verbose=0)
    eval_time = (time.time() - eval_start) / len(data.X_test) * 1000  # ms per sample

    y_pred_probs = model.predict(data.X_test, batch_size=256, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(data.y_test, y_pred).tolist()
    report = classification_report(data.y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)

    # Save upgraded model
    model_path = MODELS_DIR / "mobilenet_cifar10.keras"
    model.save(model_path)
    print(f"\nSaved upgraded fine-tuned model to: {model_path}")

    # Shipped Dense baseline from Part 15 / 16
    dense_baseline = {
        "name": "Medium Dense (3x512, Dropout 0.3)",
        "params": 2_103_818,
        "test_accuracy": 0.5408,
        "test_loss": 1.2815,
        "test_f1": 0.5399,
    }

    results = {
        "model": "MobileNetV2 (ImageNet Pre-trained + Augmented Cosine Fine-Tuning)",
        "total_epochs": WARMUP_EPOCHS + FINETUNE_EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "finetune_epochs": FINETUNE_EPOCHS,
        "batch_size": BATCH_SIZE,
        "training_time_seconds": round(total_train_time, 2),
        "total_params": int(total_params),
        "trainable_params_phase1": int(trainable_params_phase1),
        "trainable_params_phase2": int(trainable_params_phase2),
        "best_val_accuracy": float(max(combined_history["val_accuracy"])),
        "min_val_loss": float(min(combined_history["val_loss"])),
        "final_train_accuracy": float(combined_history["accuracy"][-1]),
        "final_val_accuracy": float(combined_history["val_accuracy"][-1]),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "latency_ms_per_sample": round(eval_time, 2),
        "history": {k: [float(v) for v in vals] for k, vals in combined_history.items()},
        "confusion_matrix": cm,
        "classification_report": report,
        "dense_baseline": dense_baseline,
        "accuracy_gain_vs_dense": round((test_acc - dense_baseline["test_accuracy"]) * 100, 2),
    }

    results_file = RESULTS_DIR / "mobilenet_results.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"Saved results summary to: {results_file}")

    print("\n" + "=" * 75)
    print("  FINAL UPGRADED COMPARISON: Dense MLP vs Optimized MobileNetV2")
    print("=" * 75)
    print(f"  Dense Baseline Test Accuracy:     {dense_baseline['test_accuracy'] * 100:.2f}% (50 epochs)")
    print(f"  MobileNetV2 Test Accuracy:        {test_acc * 100:.2f}% ({WARMUP_EPOCHS + FINETUNE_EPOCHS} epochs)")
    print(f"  Net Accuracy Improvement:         +{results['accuracy_gain_vs_dense']:.2f}% percentage points")
    print(f"  Weighted F1 Score:                {results['weighted_f1'] * 100:.2f}%")
    print(f"  Total Training Time:              {total_train_time:.1f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()
