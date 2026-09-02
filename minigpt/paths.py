"""Where everything lives on disk.

Notebooks are executed from ``notebooks/`` but the data and checkpoints live at
the repository root, so we resolve everything from the location of this file.
"""

from __future__ import annotations

from pathlib import Path

#: The repository root (the folder that contains ``minigpt/``).
ROOT = Path(__file__).resolve().parent.parent

#: Cleaned + raw corpora produced by ``01_data.ipynb``.
DATA = ROOT / "data"

#: Model weights (``*.pt``) produced by ``02``/``04``/``05`` and the bake script.
CHECKPOINTS = ROOT / "checkpoints"

#: Loss histories (``*.json``) used by the pre-baked path in ``03_diagnosis``.
ARTIFACTS = ROOT / "artifacts"

#: Small text files that ship with the repo so we never need the network.
ASSETS = ROOT / "minigpt" / "assets"


def ensure_dirs() -> None:
    """Create the data/checkpoint/artifact folders if they are missing."""
    for path in (DATA, CHECKPOINTS, ARTIFACTS):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "ARTIFACTS",
    "ASSETS",
    "CHECKPOINTS",
    "DATA",
    "ROOT",
    "ensure_dirs",
]
