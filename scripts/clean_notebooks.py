"""Strip outputs and execution counts from the committed notebooks.

    python scripts/clean_notebooks.py

Run this before committing. Notebooks with outputs produce enormous, unreadable
diffs, and the outputs are all reproducible anyway.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"


def clean(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for index, cell in enumerate(payload.get("cells", [])):
        # nbformat >= 4.5 requires every cell to carry a stable id.
        if not cell.get("id"):
            cell["id"] = f"cell-{index:02d}"
            changed = True
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
        cell.get("metadata", {}).pop("execution", None)
    if changed:
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        print(f"{'cleaned' if clean(path) else 'already clean'}: {path.name}")


if __name__ == "__main__":
    main()
