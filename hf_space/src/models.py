"""Model construction and rule-checking for the capstone.

Two jobs:

1. :func:`build_dense_model` builds the fully connected networks for Parts 3-15
   from a layer spec, so both of us produce structurally identical models and
   the only thing that varies between experiments is the thing we meant to vary.
2. :func:`validate_model` checks a model against the project's hard rules and
   raises if it breaks one. Run it on every model, including hand-built ones.

The rule it exists for: ``BatchNormalization`` normalises over the LAST axis.
Our data is (N, 32, 32, 3), so a BatchNorm placed BEFORE ``Flatten`` normalises
the 3 colour channels (12 parameters) instead of the 3072 features (12,288).
That mistake raises no error in Keras, trains happily, and reports plausible
accuracy -- it is simply a different model from the one Part 13 asks about, and
it would silently corrupt the Part 14 ablation. validate_model catches it.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .config import IMG_SHAPE, INPUT_DIM, NUM_CLASSES, SEED
from .utils import set_seed

__all__ = ["build_dense_model", "validate_model", "ProjectRuleError"]


class ProjectRuleError(AssertionError):
    """Raised when a model breaks one of the capstone's stated rules."""


# Layer types the brief forbids outright: "Do not use CNN, Conv2D, pretrained
# networks, or Transfer Learning."
_FORBIDDEN_SUBSTRINGS = ("Conv", "Pooling", "LSTM", "GRU", "RNN", "Attention", "Embedding")


def build_dense_model(
    hidden_units: Sequence[int],
    *,
    dropout: float | Sequence[float] | None = None,
    l2: float | None = None,
    batchnorm: bool = False,
    learning_rate: float = 1e-3,
    optimizer: str = "adam",
    seed: int | None = SEED,
    name: str | None = None,
):
    """Build and compile a fully connected network on the project's data shape.

    Each hidden block is ``Dense(relu) -> [BatchNorm] -> [Dropout]``. Toggling
    ``batchnorm`` therefore adds exactly one layer per block and changes nothing
    else, which is what makes the Part 14 ablation a clean comparison.

    Args:
        hidden_units: Neurons per hidden layer, e.g. ``[512]`` for the shallow
            model in Part 4, ``[512, 256, 128]`` for the medium one.
        dropout: Rate applied after every hidden block, or one rate per block.
            None disables it.
        l2: L2 regularisation strength applied to every hidden ``kernel``.
        batchnorm: Insert BatchNormalization after each hidden activation.
            Always lands after Flatten, so it normalises 3072 features.
        learning_rate: Passed to the optimizer.
        optimizer: ``"adam"`` or ``"sgd"`` (Part 11).
        seed: Reseeded before construction so weights start identical across
            runs. Pass None to skip (you almost never want to).
        name: Optional model name, shown in ``model.summary()``.

    Returns:
        A compiled ``keras.Sequential`` ready for ``.fit``, already checked by
        :func:`validate_model`.
    """
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    if not hidden_units:
        raise ValueError("hidden_units must contain at least one layer")

    rates = _per_layer(dropout, len(hidden_units), "dropout")
    if seed is not None:
        set_seed(seed)

    reg = regularizers.l2(l2) if l2 else None

    net: list = [
        keras.Input(shape=IMG_SHAPE),
        # Flatten FIRST. Everything after it sees (batch, 3072), which is what
        # makes a later BatchNormalization normalise features, not channels.
        layers.Flatten(name="flatten"),
    ]
    for i, (units, rate) in enumerate(zip(hidden_units, rates), start=1):
        net.append(layers.Dense(units, activation="relu", kernel_regularizer=reg, name=f"dense_{i}"))
        if batchnorm:
            net.append(layers.BatchNormalization(name=f"batchnorm_{i}"))
        if rate:
            net.append(layers.Dropout(rate, name=f"dropout_{i}"))
    net.append(layers.Dense(NUM_CLASSES, activation="softmax", name="output"))

    model = keras.Sequential(net, name=name)
    model.compile(
        optimizer=_optimizer(optimizer, learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    validate_model(model)
    return model


def validate_model(model) -> None:
    """Raise :class:`ProjectRuleError` if a model breaks a project rule.

    Checks, in order:

    * no convolutional / recurrent / pretrained layers (the brief forbids them)
    * every BatchNormalization sits after Flatten, i.e. normalises 3072
      features rather than 3 colour channels
    * the output layer has exactly 10 units

    Safe to call on any Keras model, including ones you build by hand.
    """
    from tensorflow.keras import layers

    problems: list[str] = []

    for layer in model.layers:
        cls = type(layer).__name__
        if any(bad in cls for bad in _FORBIDDEN_SUBSTRINGS):
            problems.append(
                f"'{layer.name}' is a {cls} layer. The brief forbids CNNs, Conv2D, "
                "pretrained networks and transfer learning -- use Dense layers only."
            )

        if isinstance(layer, layers.BatchNormalization):
            rank = len(layer.input.shape)
            if rank != 2:
                params = layer.count_params()
                problems.append(
                    f"'{layer.name}' receives a rank-{rank} input {tuple(layer.input.shape)}, so it "
                    f"normalises {params // 4} statistic(s) instead of {INPUT_DIM}. "
                    "BatchNormalization normalises the LAST axis -- move it AFTER "
                    "Flatten(). This does not raise in Keras but is a different model."
                )

    out = model.layers[-1]
    units = getattr(out, "units", None)
    if units is not None and units != NUM_CLASSES:
        problems.append(
            f"output layer '{out.name}' has {units} units; CIFAR-10 needs {NUM_CLASSES} "
            "(one per class)."
        )

    if problems:
        raise ProjectRuleError(
            "Model breaks project rules:\n" + "\n".join(f"  - {p}" for p in problems)
        )


def _per_layer(value, n: int, label: str) -> list:
    """Broadcast a scalar to n layers, or validate a per-layer sequence."""
    if value is None:
        return [None] * n
    if isinstance(value, (int, float)):
        return [value] * n
    values = list(value)
    if len(values) != n:
        raise ValueError(f"{label} has {len(values)} entries but there are {n} hidden layers")
    return values


def _optimizer(name: str, learning_rate: float):
    from tensorflow import keras

    key = name.lower()
    if key == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    if key == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate)
    raise ValueError(f"unknown optimizer {name!r}; use 'adam' or 'sgd' (Part 11 compares these)")
