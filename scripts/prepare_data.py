"""Build every dataset the notebooks need, without opening a notebook.

    python scripts/prepare_data.py

Notebook ``01_data.ipynb`` walks through exactly the same steps, one at a time,
with explanations.  This script is the "just give me the files" version, and it
is what ``scripts/pretrain_all.py`` calls first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minigpt import data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="tinytales",
        choices=["tinytales", "shakespeare"],
        help="tinytales = bundled + offline (default). shakespeare = ~1MB download.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data.prepare_all(source=args.source, seed=args.seed)


if __name__ == "__main__":
    main()
