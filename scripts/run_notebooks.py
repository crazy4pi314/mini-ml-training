"""Execute every notebook end to end and report how long each one took.

    python scripts/run_notebooks.py
    python scripts/run_notebooks.py 02_pretrain.ipynb 03_diagnosis.ipynb

Notebooks are executed **in place in a temporary copy**, so the committed files
keep their (empty) outputs.  A non-zero exit code means at least one notebook
raised.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"


def run_one(path: Path, timeout: int) -> tuple[bool, float, str]:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOKS)}},
    )
    started = time.time()
    try:
        client.execute()
    except CellExecutionError as exc:
        return False, time.time() - started, str(exc)[-2000:]
    return True, time.time() - started, ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="notebook filenames (default: all)")
    parser.add_argument("--timeout", type=int, default=1800, help="per-cell timeout, seconds")
    args = parser.parse_args()

    names = args.names or sorted(p.name for p in NOTEBOOKS.glob("*.ipynb"))
    results: list[tuple[str, bool, float]] = []

    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            source = NOTEBOOKS / name
            scratch = Path(tmp) / name
            shutil.copy2(source, scratch)
            print(f"--- running {name} ...", flush=True)
            ok, seconds, error = run_one(scratch, args.timeout)
            results.append((name, ok, seconds))
            print(f"    {'OK ' if ok else 'FAIL'} {seconds:6.1f}s", flush=True)
            if not ok:
                print(error)

    print("\n" + "=" * 60)
    print(f"{'notebook':<34}{'result':>8}{'seconds':>10}")
    print("-" * 60)
    for name, ok, seconds in results:
        print(f"{name:<34}{'OK' if ok else 'FAIL':>8}{seconds:>10.1f}")
    total = sum(s for _, _, s in results)
    print("-" * 60)
    print(f"{'TOTAL':<34}{'':>8}{total:>10.1f}")

    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
