"""Shared code for the Deep Learning capstone.

Keeping data loading, seeding and model building here (rather than copy-pasted
into each notebook) is what lets two people run genuinely comparable
experiments. Start with the notebook in Notebooks/ covering Parts 1-2.

    from src.data import load_splits      # the ONLY way to get the data
    from src.models import build_dense_model, validate_model
    from src.utils import set_seed

Requires Python 3.10 or newer and the packages in Docs/requirements.txt.
Runs on Google Colab as well as a local environment.
"""

import sys

# 3.10 is the real floor: it is the lowest version every range in
# Docs/requirements.txt resolves on, and nothing here uses 3.11+ syntax.
# Keeping the guard at the true minimum matters because Colab pins its own
# interpreter -- refusing to run on a version that works would block a
# teammate for no reason.
if sys.version_info < (3, 10):
    raise RuntimeError(
        f"This project needs Python 3.10 or newer, but you are running "
        f"{sys.version_info.major}.{sys.version_info.minor}.\n"
        "Create a fresh environment with a supported interpreter:\n"
        "    py -3.12 -m venv .venv          (Windows)\n"
        "    python3.12 -m venv .venv        (macOS / Linux)\n"
        "then activate it and run: pip install -r Docs/requirements.txt"
    )
