"""Single source of truth for the CIFAR-10 data used in this project.

Every notebook -- from the baseline in Part 3 to the final test evaluation in
Part 16 -- must get its arrays from this module. That is what makes the
comparisons in Parts 5, 7, 14 and 18 fair, and what keeps two people working on
separate branches from silently training on different data.

Typical use:

    from src.data import load_splits, class_names
    d = load_splits()
    d.X_train.shape  # (45000, 32, 32, 3)

Models start with ``keras.layers.Flatten()``, which turns each (32, 32, 3)
image into the 3072-value vector the Dense layers need. See load_splits().

The split itself lives in Dataset/splits.npz, which IS committed to git. The
raw CIFAR-10 download (~170 MB) is cached by Keras in ~/.keras/datasets and is
NOT committed.
"""

import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import (
    CIFAR_BATCH_FILES,
    CIFAR_LOCAL_DIR,
    CLASS_NAMES,
    INPUT_DIM,
    IMG_SHAPE,
    NUM_CLASSES,
    N_TRAIN_FULL,
    N_VAL,
    SEED,
    SPLIT_FILE,
)

__all__ = [
    "Splits",
    "load_raw",
    "load_splits",
    "make_split_indices",
    "class_names",
    "to_onehot",
]


@dataclass(frozen=True)
class Splits:
    """The six arrays required by Part 2, plus the raw images for plotting.

    X_* are float32 in [0, 1]. y_* are int32 class indices with shape (N,) --
    use ``sparse_categorical_crossentropy``, or call :func:`to_onehot` if you
    prefer ``categorical_crossentropy``.

    images_val and images_test keep the original (N, 32, 32, 3) uint8 pixels so
    misclassified images can be displayed in Part 17. Never feed them to a
    model. There is deliberately no images_train: Part 1 displays training
    samples straight from :func:`load_raw`, so keeping a second 138 MB copy of
    them here would be pure waste.
    """

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    images_val: np.ndarray
    images_test: np.ndarray

    def summary(self) -> str:
        """Formatted shape report -- paste the output into the Part 2 write-up."""
        rows = [
            ("X_train", self.X_train),
            ("X_val", self.X_val),
            ("X_test", self.X_test),
            ("y_train", self.y_train),
            ("y_val", self.y_val),
            ("y_test", self.y_test),
        ]
        # Widths measured, not hard-coded: the X shapes are 19 characters wide
        # under the default (N, 32, 32, 3) but only 13 under flatten=True.
        name_w = max(len(name) for name, _ in rows)
        shape_w = max(len(str(arr.shape)) for _, arr in rows)
        return "\n".join(
            f"{name:<{name_w}}  {str(arr.shape):<{shape_w}}  {arr.dtype}" for name, arr in rows
        )


def class_names() -> list[str]:
    """The 10 CIFAR-10 class names, ordered so index i is the label for class i."""
    return list(CLASS_NAMES)


CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _keras_cache_dir() -> Path:
    """Where Keras caches downloaded datasets (honours the KERAS_HOME env var)."""
    return Path(os.environ.get("KERAS_HOME", Path.home() / ".keras")) / "datasets"


@contextmanager
def _quiet_unpickle():
    """Silence one specific NumPy warning while CIFAR-10 batch files are read.

    The batch files were pickled in 2009 and rebuild a dtype with align=0 -- an
    int where NumPy >=2.4 wants a bool -- so every unpickle emits a
    VisibleDeprecationWarning. The data is unaffected (verified byte-identical
    to keras.datasets.cifar10.load_data), and there is nothing to fix on our
    side: the warning comes from the file format itself.

    Both loading paths need this. Ours calls it around pickle.load; the Keras
    fallback needs it too, because Keras unpickles the same files inside
    keras/src/datasets/cifar.py and emits six identical warnings.

    Matched on the message so this hides exactly that one line -- any other
    warning, including other DeprecationWarnings, still reaches you.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"dtype\(\): align should be passed",
            category=UserWarning,  # VisibleDeprecationWarning subclasses UserWarning
        )
        yield


def _unzip_instructions() -> str:
    """The single place the 'use the shared copy' instruction is written."""
    return (
        f"Ask your teammate for '{CIFAR_LOCAL_DIR.name}.zip' and unzip it so the\n"
        "batch files sit directly in this folder, with no extra nesting:\n"
        f"  {CIFAR_LOCAL_DIR}\n"
        f"    {', '.join(CIFAR_BATCH_FILES)}\n"
    )


def _download_help(cache: Path) -> str:
    return (
        "The CIFAR-10 download did not complete.\n"
        "\n"
        "FASTEST FIX -- no download at all:\n"
        f"{_unzip_instructions()}"
        "\n"
        "OR resume the download. cs.toronto.edu serves at roughly 115 KB/s, so the\n"
        "170 MB archive takes ~25 minutes, and Keras's downloader cannot resume a\n"
        "partial file. curl can -- re-run this until it completes:\n"
        "\n"
        f'    curl -L -C - -o "{cache / "cifar-10-batches-py-target_archive"}" \\\n'
        f"      {CIFAR_URL}\n"
        "\n"
        "Then re-run this cell. Keras verifies the checksum and extracts it."
    )


def _local_data_ready() -> bool:
    """True if Dataset/cifar-10-batches-py holds a complete copy of the dataset.

    Only that exact path is accepted -- no searching and no unwrapping of nested
    folders, so there is one correct layout rather than several that happen to
    work. A folder that exists but is incomplete raises instead of silently
    falling back to a 25-minute download.
    """
    if not CIFAR_LOCAL_DIR.is_dir():
        return False

    missing = [f for f in CIFAR_BATCH_FILES if not (CIFAR_LOCAL_DIR / f).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{CIFAR_LOCAL_DIR}\nexists but is missing {len(missing)} of "
            f"{len(CIFAR_BATCH_FILES)} required files: {', '.join(missing)}\n"
            "\n"
            f"{_unzip_instructions()}"
            "\n"
            "Delete the folder entirely if you would rather download the dataset."
        )
    return True


def _load_batch(fpath: Path):
    """Read one CIFAR-10 pickle batch into (N, 3, 32, 32) uint8 + labels.

    Mirrors keras.src.datasets.cifar.load_batch. Reimplemented rather than
    imported because that module is private API, and this way the local path
    does not break if Keras reorganises its internals.
    """
    import pickle

    with _quiet_unpickle():
        with open(fpath, "rb") as f:
            d = pickle.load(f, encoding="bytes")

    data = d[b"data"]
    labels = d[b"labels"]
    return data.reshape(data.shape[0], 3, 32, 32), np.array(labels, dtype="uint8")


def _load_local():
    """Assemble the full dataset from the batch files in CIFAR_LOCAL_DIR.

    Batch order matters: the committed split in Dataset/splits.npz indexes into
    data_batch_1..5 concatenated in that order, exactly as Keras assembles them.
    """
    x_train = np.empty((N_TRAIN_FULL, 3, 32, 32), dtype="uint8")
    y_train = np.empty((N_TRAIN_FULL,), dtype="uint8")
    for i in range(1, 6):
        lo, hi = (i - 1) * 10_000, i * 10_000
        x_train[lo:hi], y_train[lo:hi] = _load_batch(CIFAR_LOCAL_DIR / f"data_batch_{i}")
    x_test, y_test = _load_batch(CIFAR_LOCAL_DIR / "test_batch")
    # channels-first on disk -> channels-last, matching cifar10.load_data()
    return x_train.transpose(0, 2, 3, 1), y_train, x_test.transpose(0, 2, 3, 1), y_test


def load_raw():
    """Return CIFAR-10 exactly as distributed: uint8 images, int labels.

    Looks for the batch files in Dataset/cifar-10-batches-py/ first. If they are
    there -- because someone unzipped the shared copy into it -- loading takes a
    second and nothing is downloaded. Otherwise falls back to Keras, which
    fetches ~170 MB from cs.toronto.edu at roughly 115 KB/s (~25 minutes) and
    caches it under ~/.keras/datasets.

    Either path produces identical arrays, so the committed split stays valid.
    """
    if _local_data_ready():
        return _prepare_raw(*_load_local())

    from tensorflow.keras.datasets import cifar10

    cache = _keras_cache_dir()
    # Warn BEFORE the download starts. Without this, a first run looks like a
    # 25-minute hang and the natural reaction is to interrupt the kernel --
    # which leaves a partial file that Keras cannot resume.
    if not (cache / "cifar-10-batches-py-target").exists():
        print(
            f"No dataset found at {CIFAR_LOCAL_DIR}\n"
            "Downloading ~170 MB from cs.toronto.edu, which serves at about 115 KB/s\n"
            "-- expect ~25 minutes. Let it run; do NOT interrupt the kernel, because\n"
            "Keras cannot resume a partial download.\n"
            "\n"
            "To skip the download entirely:\n"
            f"{_unzip_instructions()}",
            flush=True,
        )

    try:
        with _quiet_unpickle():
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    except Exception as exc:  # noqa: BLE001 - re-raised with actionable guidance
        raise RuntimeError(f"{_download_help(cache)}\n\nOriginal error: {exc!r}") from exc

    return _prepare_raw(x_train, y_train, x_test, y_test)


def _prepare_raw(x_train, y_train, x_test, y_test):
    """Normalise label shape/dtype across both loading paths.

    Keras hands back labels shaped (N, 1); flatten to (N,) so indexing and
    sklearn metrics behave predictably.
    """
    return (
        x_train,
        np.asarray(y_train).ravel().astype("int32"),
        x_test,
        np.asarray(y_test).ravel().astype("int32"),
    )


def make_split_indices(y_train_full: np.ndarray, n_val: int = N_VAL, seed: int = SEED):
    """Build a stratified train/validation split of the 50,000 training images.

    Stratified means each of the 10 classes contributes exactly n_val / 10
    images to the validation set, so validation accuracy is not distorted by an
    unlucky class imbalance.
    """
    if n_val % NUM_CLASSES:
        raise ValueError(f"n_val={n_val} must be divisible by {NUM_CLASSES} for an even split")

    per_class = n_val // NUM_CLASSES
    rng = np.random.default_rng(seed)

    val_idx = []
    for label in range(NUM_CLASSES):
        (class_idx,) = np.nonzero(y_train_full == label)
        val_idx.append(rng.choice(class_idx, size=per_class, replace=False))
    val_idx = np.sort(np.concatenate(val_idx))

    mask = np.ones(len(y_train_full), dtype=bool)
    mask[val_idx] = False
    (train_idx,) = np.nonzero(mask)

    # Shuffle so neither set is ordered by class; Keras's own validation_split
    # would not do this, which is one reason we build the split ourselves.
    train_idx = rng.permutation(train_idx)
    val_idx = rng.permutation(val_idx)
    return train_idx.astype("uint16"), val_idx.astype("uint16")


def _get_split_indices(y_train_full: np.ndarray):
    """Load the committed split, creating it on first run."""
    if SPLIT_FILE.exists():
        with np.load(SPLIT_FILE) as f:
            train_idx, val_idx = f["train_idx"], f["val_idx"]
    else:
        train_idx, val_idx = make_split_indices(y_train_full)
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(SPLIT_FILE, train_idx=train_idx, val_idx=val_idx, seed=SEED)
        print(f"Created {SPLIT_FILE.name} -- commit this file so we share the same split.")

    # Cheap guard against a corrupted or hand-edited split file.
    n_total = len(train_idx) + len(val_idx)
    if n_total != N_TRAIN_FULL or len(np.union1d(train_idx, val_idx)) != N_TRAIN_FULL:
        raise RuntimeError(
            f"{SPLIT_FILE} is invalid: got {n_total} indices covering "
            f"{len(np.union1d(train_idx, val_idx))} unique positions. "
            "Delete the file and re-run to regenerate it."
        )
    return train_idx.astype("int64"), val_idx.astype("int64")


def load_splits(flatten: bool = False, scale: bool = True) -> Splits:
    """Return the project's train / validation / test split.

    Args:
        flatten: Leave False (the project default). X arrays keep their
            (N, 32, 32, 3) image shape and every model starts with a
            ``keras.layers.Flatten()`` layer, which is the conversion Part 2
            requires and which shows up as a row in ``model.summary()``.
            Passing True instead returns pre-flattened (N, 3072) rows for a
            model built with ``keras.Input(shape=(3072,))``. The two are
            numerically identical -- same parameter count, same metrics to the
            last decimal -- so do not mix them across notebooks.

            One trap when building models on the default (N, 32, 32, 3) shape:
            ``BatchNormalization`` normalises over the LAST axis, so it must
            come *after* Flatten. Before Flatten it normalises the 3 colour
            channels (12 parameters) instead of the 3072 features (12,288).
            That mistake raises no error -- check model.summary().
        scale: Divide pixels by 255 to land in [0, 1]. Leave this True; the
            argument exists only so you can demonstrate in Part 1 that the raw
            range is 0-255.

    Holds roughly 700 MB of RAM: the float32 copies dominate.
    """
    x_train_full, y_train_full, x_test, y_test = load_raw()
    train_idx, val_idx = _get_split_indices(y_train_full)

    images_val = x_train_full[val_idx]
    y_train = y_train_full[train_idx]
    y_val = y_train_full[val_idx]

    def prepare(images: np.ndarray) -> np.ndarray:
        # ascontiguousarray, not astype: Keras hands back a transposed view of
        # channel-first data, and astype's default order="K" would faithfully
        # preserve that strided layout. TensorFlow then copies every batch into
        # C order on the fly. Paying for one copy here is cheaper, and it keeps
        # flatten=True and flatten=False on equal footing -- the reshape in the
        # flatten=True branch happens to force a copy anyway.
        arr = np.ascontiguousarray(images, dtype="float32")
        if scale:
            arr /= 255.0
        return arr.reshape(len(arr), INPUT_DIM) if flatten else arr.reshape((-1, *IMG_SHAPE))

    return Splits(
        X_train=prepare(x_train_full[train_idx]),
        X_val=prepare(images_val),
        X_test=prepare(x_test),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        # Contiguous for the same reason as the X arrays: Keras's loader hands
        # back a transposed view, and matplotlib re-copies a strided array on
        # every imshow call during the Part 17 error analysis.
        images_val=np.ascontiguousarray(images_val),
        images_test=np.ascontiguousarray(x_test),
    )


def to_onehot(y: np.ndarray) -> np.ndarray:
    """(N,) integer labels -> (N, 10) one-hot, for categorical_crossentropy."""
    return np.eye(NUM_CLASSES, dtype="float32")[y]
