"""Reproducibility helpers.

Part 5 requires the same random seed across the architecture comparison, and
Part 9 requires each learning-rate run to start from fresh weights. Call
:func:`set_seed` immediately before building every model so weight
initialisation, shuffling and dropout masks are identical between runs -- and
between the two of us.
"""

import os
import random

import numpy as np

from .config import SEED

__all__ = ["set_seed"]


def set_seed(seed: int = SEED, deterministic: bool = False) -> None:
    """Seed Python, NumPy and TensorFlow in one call.

    Args:
        seed: Defaults to the project seed in config.py.
        deterministic: Also force deterministic GPU kernels. Guarantees
            bit-identical results at some cost in speed; unnecessary on CPU.
    """
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if deterministic:
        tf.config.experimental.enable_op_determinism()
