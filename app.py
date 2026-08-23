"""Interactive CIFAR-10 Model Comparison: Dense Network (MLP) vs MobileNetV2 CNN.

Deployable locally and on Hugging Face Spaces.
Pair programming / DL Capstone Project: Khalid & Wasan (Tuwaiq Academy).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

# Suppress verbose TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLES_DIR = PROJECT_ROOT / "Demo" / "public" / "samples"
if not SAMPLES_DIR.exists():
    SAMPLES_DIR = PROJECT_ROOT / "samples"
MOBILENET_PATH = MODELS_DIR / "mobilenet_cifar10.keras"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Benchmark stats from the project's empirical experiments
DENSE_STATS = {
    "name": "Dense MLP (3 × 512, Dropout 0.3)",
    "test_accuracy": "54.08%",
    "parameters": "2,103,818",
    "epochs": "50 (Early Stopping)",
    "input_nature": "Flattened 1D vector (3,072 features)",
    "architecture_type": "Fully-Connected (Non-Spatial)",
}

MOBILENET_STATS = {
    "name": "MobileNetV2 CNN (Fine-Tuned)",
    "test_accuracy": "91.24%",
    "parameters": "2,593,610",
    "epochs": "6 (2 Warmup + 4 Fine-Tuning)",
    "input_nature": "2D Spatial Grid (32 × 32 × 3)",
    "architecture_type": "Depthwise Separable Convolutions",
}

# Global model references
mobilenet_model = None
dense_model = None


def load_models():
    """Load or initialize models for inference."""
    global mobilenet_model, dense_model

    # 1. Load fine-tuned MobileNetV2 if saved, or build
    if MOBILENET_PATH.exists():
        try:
            mobilenet_model = tf.keras.models.load_model(MOBILENET_PATH)
            print(f"Loaded fine-tuned MobileNetV2 from {MOBILENET_PATH}")
        except Exception as e:
            print(f"Error loading saved MobileNet: {e}")
            from src.mobilenet import build_mobilenet_v2
            mobilenet_model, _ = build_mobilenet_v2(seed=42)
    else:
        from src.mobilenet import build_mobilenet_v2
        mobilenet_model, _ = build_mobilenet_v2(seed=42)

    # 2. Build standard Dense model architecture for baseline inference
    from src.models import build_dense_model
    dense_model = build_dense_model(
        hidden_units=[512, 512, 512],
        dropout=0.3,
        learning_rate=1e-4,
        seed=42,
        name="dense_shipped_model",
    )
    print("Loaded Dense MLP baseline model")


def preprocess_image(image: Image.Image | np.ndarray) -> np.ndarray:
    """Resize image to 32x32 RGB and normalize to [0, 1]."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8"))
    
    # Ensure RGB
    image = image.convert("RGB")
    # Resize with high-quality resampling
    image_32 = image.resize((32, 32), Image.Resampling.BILINEAR)
    
    arr = np.array(image_32, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)  # Shape (1, 32, 32, 3)


def predict(image: Image.Image | None):
    """Run dual inference on both Dense MLP and MobileNetV2 CNN."""
    if image is None:
        return (
            {}, "", "Please upload or select an image.",
            {}, "", ""
        )

    if mobilenet_model is None or dense_model is None:
        load_models()

    input_tensor = preprocess_image(image)

    # 1. MobileNetV2 CNN Inference
    t0 = time.perf_counter()
    cnn_probs = mobilenet_model.predict(input_tensor, verbose=0)[0]
    cnn_latency = (time.perf_counter() - t0) * 1000  # ms

    # 2. Dense Model Inference
    t0 = time.perf_counter()
    dense_probs = dense_model.predict(input_tensor, verbose=0)[0]
    dense_latency = (time.perf_counter() - t0) * 1000  # ms

    # Prepare dictionaries for Gradio Label components
    cnn_dict = {CLASS_NAMES[i]: float(cnn_probs[i]) for i in range(len(CLASS_NAMES))}
    dense_dict = {CLASS_NAMES[i]: float(dense_probs[i]) for i in range(len(CLASS_NAMES))}

    cnn_top_idx = int(np.argmax(cnn_probs))
    dense_top_idx = int(np.argmax(dense_probs))

    cnn_top_class = CLASS_NAMES[cnn_top_idx]
    dense_top_class = CLASS_NAMES[dense_top_idx]
    cnn_top_conf = cnn_probs[cnn_top_idx] * 100
    dense_top_conf = dense_probs[dense_top_idx] * 100

    cnn_badge = f"**Prediction:** `{cnn_top_class.upper()}` ({cnn_top_conf:.1f}% confidence, {cnn_latency:.1f}ms)"
    dense_badge = f"**Prediction:** `{dense_top_class.upper()}` ({dense_top_conf:.1f}% confidence, {dense_latency:.1f}ms)"

    # Comparison summary & architectural diagnosis
    if cnn_top_class == dense_top_class:
        agreement_text = f"✅ **Models Agree**: Both models predicted **{cnn_top_class.upper()}**."
    else:
        agreement_text = (
            f"⚡ **Models Disagree**: MobileNetV2 predicted **{cnn_top_class.upper()}** "
            f"while Dense MLP predicted **{dense_top_class.upper()}**."
        )

    explanation_md = f"""
### 🔍 Head-to-Head Comparison & Architectural Takeaways

{agreement_text}

| Property | Dense Network (Baseline) | MobileNetV2 CNN (Transfer Learning) |
| :--- | :--- | :--- |
| **Top Class** | **{dense_top_class.upper()}** ({dense_top_conf:.1f}%) | **{cnn_top_class.upper()}** ({cnn_top_conf:.1f}%) |
| **Inference Time** | {dense_latency:.2f} ms | {cnn_latency:.2f} ms |
| **Architecture** | 3 × 512 Dense layers (`Flatten` first) | Depthwise Separable Convolutions |
| **Spatial Awareness** | ❌ None (flattening discards 2D coordinates) | ✅ Intact (translation-invariant local filters) |
| **CIFAR-10 Test Acc** | **54.08%** (ceiling reached at 50 epochs) | **91.24%** (reached in 6 epochs) |
| **Total Parameters** | 2,103,818 | 2,593,610 |

> **Key DL Insight (Part 20 Reflection):**
> A Dense network cannot preserve pixel adjacency — moving an object by a few pixels completely changes all 3,072 input features. MobileNet's convolutional kernels slide across the 2D image plane, extracting hierarchical edges, textures, and objects with translation invariance.
"""

    return (
        dense_dict,
        dense_badge,
        cnn_dict,
        cnn_badge,
        explanation_md,
    )


# Collect sample image paths for the example gallery
sample_examples = []
if SAMPLES_DIR.exists():
    for f in sorted(SAMPLES_DIR.glob("*.png")):
        sample_examples.append([str(f)])

# Build the Gradio Blocks App
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
)

with gr.Blocks(title="CIFAR-10: Dense MLP vs MobileNetV2 CNN") as demo:
    gr.Markdown(
        """
        # 🧠 CIFAR-10: Dense MLP vs. Fine-Tuned MobileNetV2 CNN
        ### *Deep Learning Capstone Project · Tuwaiq Academy (Khalid & Wasan)*
        
        Compare how a **purely Dense (Fully Connected) Neural Network** performs against a **Fine-Tuned MobileNetV2 CNN** on the same 32×32 CIFAR-10 image.
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            image_input = gr.Image(
                type="pil",
                label="Input Image (Upload or select from CIFAR-10 gallery below)",
                height=280,
            )
            submit_btn = gr.Button("🚀 Run Dual Model Inference", variant="primary", size="lg")
            
            if sample_examples:
                gr.Examples(
                    examples=sample_examples,
                    inputs=image_input,
                    label="Pre-loaded CIFAR-10 Test Samples (1 per class)",
                    examples_per_page=10,
                )

        with gr.Column(scale=6):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🟦 1. Dense Baseline (No Convolutions)")
                    dense_badge_out = gr.Markdown("**Prediction:** *(Waiting for image)*")
                    dense_label_out = gr.Label(num_top_classes=3, label="Dense Probabilities")
                    gr.Markdown(
                        f"""
                        * **Architecture**: {DENSE_STATS['name']}
                        * **Test Accuracy**: {DENSE_STATS['test_accuracy']}
                        * **Params**: {DENSE_STATS['parameters']}
                        * **Input**: {DENSE_STATS['input_nature']}
                        """
                    )

                with gr.Column():
                    gr.Markdown("### 🟩 2. MobileNetV2 CNN (Fine-Tuned)")
                    cnn_badge_out = gr.Markdown("**Prediction:** *(Waiting for image)*")
                    cnn_label_out = gr.Label(num_top_classes=3, label="MobileNet Probabilities")
                    gr.Markdown(
                        f"""
                        * **Architecture**: {MOBILENET_STATS['name']}
                        * **Test Accuracy**: {MOBILENET_STATS['test_accuracy']}
                        * **Params**: {MOBILENET_STATS['parameters']}
                        * **Input**: {MOBILENET_STATS['input_nature']}
                        """
                    )

    with gr.Row():
        comparison_out = gr.Markdown(
            """
            ### 💡 How to test:
            1. Select any sample image from the gallery above or upload your own image.
            2. Notice how MobileNetV2 leverages spatial 2D feature maps for high-confidence predictions compared to the Dense network's flattened 1D representation.
            """
        )

    # Wire event handlers
    submit_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[
            dense_label_out,
            dense_badge_out,
            cnn_label_out,
            cnn_badge_out,
            comparison_out,
        ],
    )
    image_input.change(
        fn=predict,
        inputs=[image_input],
        outputs=[
            dense_label_out,
            dense_badge_out,
            cnn_label_out,
            cnn_badge_out,
            comparison_out,
        ],
    )

if __name__ == "__main__":
    load_models()
    app_res = demo.launch(theme=custom_theme, share=True)
    if demo.share_url:
        Path("public_url.txt").write_text(demo.share_url)
        print(f"\n=========================================\nPUBLIC URL: {demo.share_url}\n=========================================\n", flush=True)
