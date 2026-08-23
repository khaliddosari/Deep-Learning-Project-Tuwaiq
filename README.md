# CIFAR-10 with Dense Networks Only

Deep Learning capstone for the Building & Developing AI Models Bootcamp at **Tuwaiq Academy**, by **Khalid** and **Wasan**.

The project classifies CIFAR-10 using fully connected networks and **no convolution**, then measures exactly how much
that constraint costs by fine-tuning a MobileNetV2 CNN on the same split. The repository holds the whole 20-part
study: the notebooks that ran it, the shared Python package that keeps every experiment comparable, the raw result
files, two deployable demos, and the write-ups.

| Model | Input | Test accuracy | Weighted F1 | Parameters |
|---|---|---|---|---|
| Dense MLP, 3 x 512 + Dropout 0.3 (shipped) | flattened 3,072 vector | **54.08%** | 53.99% | 2,103,818 |
| MobileNetV2, ImageNet pretrained + fine-tuned | 32 x 32 x 3 upsampled to 96 x 96 | **91.24%** | 91.20% | 2,593,610 |

That is a **37.16 point** gap on identical data, and it is the project's central result: the ceiling was architectural,
not a hyperparameter that went untuned.

## The constraint

For Parts 1 to 19 there is no CNN, no `Conv2D`, no pretrained weights, no transfer learning. Every image is flattened
from 32 x 32 x 3 into a 3,072 value vector before it reaches the first layer, which destroys pixel adjacency.

This is enforced in code, not by convention. `validate_model` in [src/models.py](src/models.py) raises on any
convolutional, recurrent or embedding layer, and on a `BatchNormalization` placed before `Flatten`, where it would
silently normalise 3 colour channels instead of 3,072 features. Run it on every model you build.

The MobileNetV2 work in [scripts/train_mobilenet.py](scripts/train_mobilenet.py) is the Part 20 extension. It exists
to quantify the constraint, and it is kept strictly outside the Dense results.

## Repository layout

| Path | What is in it |
|---|---|
| [Notebooks/](Notebooks/) | The 20 parts, in three notebooks. Run them in filename order. |
| [src/](src/) | Shared package. Config, data loading, model builders, rule checks, seeding. |
| [scripts/](scripts/) | Standalone experiments and training runs that live outside the notebooks. |
| [Results/](Results/) | Raw JSON produced by those runs. Every figure quoted anywhere traces back here. |
| [Dataset/](Dataset/) | `splits.npz`, the committed train/val split. Raw CIFAR-10 batches are not committed. |
| [Demo/](Demo/) | React web app presenting the full study in four tabs. See [Demo/README.md](Demo/README.md). |
| [hf_space/](hf_space/) | Hugging Face Space packaging of the Gradio app. See [hf_space/README.md](hf_space/README.md). |
| [app.py](app.py) | Gradio app comparing both models side by side, runnable locally. |
| [Docs/](Docs/) | `requirements.txt`, the capstone report (`.docx`) and the presentation (`.pptx`). |

## The 20 parts

| Parts | Notebook | Covers |
|---|---|---|
| 1 to 2 | [Dataset understanding and preparation](Notebooks/01-02_dataset_understanding_and_preparation.ipynb) | Dataset facts, class balance, pixel range, normalization, and the stratified 45,000 / 5,000 / 10,000 split. |
| 3 to 8 | [Baseline and architectures](Notebooks/03-08_baseline_and_architectures.ipynb) | The baseline network and why its hyperparameters were chosen, the shallow / medium / deep comparison under a fair protocol, learning curves, and the overfitting diagnosis. |
| 9 to 20 | [Optimization and final analysis](Notebooks/9_20_optimization_and_final_analysis.ipynb) | Learning rate, batch size and optimizer sweeps, four regularizers, training stability, the Dropout ablation, the final model, the sealed test evaluation, augmentation, per-class analysis, conclusions and reflection. |

## Setup

Python 3.11 to 3.13 (the version ranges in `Docs/requirements.txt` need 3.11 or newer).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r Docs/requirements.txt
```

CIFAR-10 itself is not committed. Keras downloads it into `~/.keras/datasets` on first use, or you can drop the
`cifar-10-batches-py` folder into `Dataset/` and it will be found there first. What **is** committed is
`Dataset/splits.npz`, so everyone trains and validates on byte-identical data.

Always load the data through the one entry point:

```python
from src.data import load_splits
from src.models import build_dense_model, validate_model
from src.utils import set_seed

d = load_splits()          # d.X_train.shape == (45000, 32, 32, 3)
set_seed(42)               # call immediately before every build
```

## Reproducing the results

Run the notebooks in order, then the scripts from the project root:

```bash
python scripts/part15_rerun.py         # the 2x2 Dropout / L2 experiment      -> Results/part15_rerun.json
python scripts/train_mobilenet.py      # 6 epochs, 2 warmup + 4 fine-tuning   -> Results/mobilenet_results.json + models/
python scripts/sample_predictions.py   # both models over the 10 demo images  -> Results/sample_predictions.json + models/dense_final.keras
```

Trained weights are **not** committed, since `.gitignore` excludes `*.keras` and GitHub rejects files over 100 MB. The
two scripts above regenerate `models/mobilenet_cifar10.keras` and `models/dense_final.keras`, which is what the Gradio
app loads. `sample_predictions.py` is the only place Dense weights get saved at all, because the notebook never saved
its own.

Seeding fixes the initial weights and the shuffle order, but not the order floating point reductions accumulate across
threads nor which kernels XLA fuses. Re-running the identical shipped configuration gave 53.81% against the notebook's
54.08%. That drift is about the size of the differences some experiments were ranking, which is why every comparison
here is made **within** one run set and never against a number from a different session.

## Web demo

A four-tab React app that walks through the whole study, with inline SVG charts drawn from the per-epoch arrays in
`Results/`. No smoothing and no interpolation, so a kink in a curve is a kink in the run.

```bash
cd Demo
npm install
npm run dev        # http://localhost:3000
npm run share      # serve on 0.0.0.0 for others on the same Wi-Fi
npm test           # build, then verify the rendered HTML and every data invariant
npm run deploy     # build and publish to Cloudflare Workers
```

Node 22.13 or newer. The test suite is the guard rail on honesty: it re-derives the confusion matrix totals, the
per-class precision, recall and F1, every run's best validation accuracy and gap, and every stated contrast in the
Part 15 re-run straight from the raw JSON. A hand-typed figure that does not match its source fails the build.
[Demo/README.md](Demo/README.md) documents where each number comes from.

## Gradio demo

Head to head inference on the two saved models, with confidence bars, latency, and the ten CIFAR-10 sample images:

```bash
python app.py
```

[hf_space/](hf_space/) is the same app packaged for Hugging Face Spaces, with a trimmed `requirements.txt` and a Space
card in its README frontmatter. Copy the trained `.keras` files into `hf_space/models/` before pushing, since they are
not tracked by git.

## What the experiments found

- **Dropout was the only regularizer that mattered.** Against an unregularized control it was worth about 2.2 points of
  validation accuracy, roughly three standard errors. L2 alone was worth 0.4 points, and L2 stacked on top of Dropout
  added 0.74 points, inside noise, so the simpler model shipped.
- **Batch Normalization failed badly**, producing the worst run in the project at 44.64% validation while reaching the
  highest training accuracy at 99.20%. Memorisation, not learning.
- **Depth stopped paying.** The shallow, medium and deep networks land close enough together that extra layers were
  rejected on the same reasoning that later rejected stacked regularizers.
- **The generalization gap fell from 32.11% to 4.47%** once Dropout and Early Stopping were in place.
- **The best score recorded anywhere was 59.12%**, using flip and shift augmentation as an extra experiment. Still more
  than 30 points below a fine-tuned CNN.
- **Two gap definitions are used and never mixed.** `GAP_PEAK` is final training accuracy minus *best* validation
  accuracy; `GAP_FINAL` uses the *final epoch* validation accuracy. Every reported gap says which one it is.

During the project proper the test set was opened exactly once, after every architecture and hyperparameter decision
was final. The Part 15 re-run opened it a second time, after its own selection had been made on validation and
recorded, with no tuning following. Both openings are reported rather than merged.

## Result files

| File | Contents |
|---|---|
| `Results/baseline.json`, `shallow.json`, `medium.json`, `deep.json` | Per-epoch curves for the Parts 4 to 7 architecture comparison. |
| `Results/part15_rerun.json` | The full 2 x 2 Dropout / L2 experiment, all four arms. |
| `Results/mobilenet_results.json` | MobileNetV2 history, confusion matrix, classification report, and the Dense comparison. |
| `Results/sample_predictions.json` | Full probability vectors from both models over the ten demo images. |

---

Khalid and Wasan, Building & Developing AI Models Bootcamp, Tuwaiq Academy. Released under the MIT license.
