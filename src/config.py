"""Project-wide constants. Import from here, never hard-code these values.

Both collaborators must use identical values for every controlled experiment
(Part 5: "Same random seed", "Same training data", "Same preprocessing").
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
RESULTS_DIR = PROJECT_ROOT / "Results"
SPLIT_FILE = DATASET_DIR / "splits.npz"  # committed to git: guarantees an identical split for both of us

# Where the CIFAR-10 batch files live if someone drops the folder in by hand.
# Checked before falling back to Keras's 25-minute download from cs.toronto.edu.
# Too big for git (178 MB) -- .gitignore excludes it; share it as a zip instead.
CIFAR_LOCAL_DIR = DATASET_DIR / "cifar-10-batches-py"
CIFAR_BATCH_FILES = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- dataset facts
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
INPUT_DIM = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS  # 3072
NUM_CLASSES = 10

# Index i == integer label i in the CIFAR-10 label encoding.
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

# ---------------------------------------------------------------- split sizes
# CIFAR-10 ships 50,000 train + 10,000 test. We carve the validation set out of
# the training half only. The test set is untouched until Part 16.
N_TRAIN_FULL = 50_000
N_VAL = 5_000                        # 10% of the training half, stratified
N_TRAIN = N_TRAIN_FULL - N_VAL       # 45,000
N_TEST = 10_000
