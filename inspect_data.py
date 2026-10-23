"""Inspect raw MAT files without modifying them; run with the SSH interpreter."""
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def main():
    root = Path(__file__).resolve().parent / "data"
    print(f"Data directory: {root}", flush=True)
    files = sorted(root.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No MAT files found in {root}")
    failures = []
    for path in files:
        print(f"\n=== {path.name} ===", flush=True)
        try:
            values = loadmat(path)
            for key, value in values.items():
                if key.startswith("__"):
                    continue
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                if not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf":
                    continue
                if value.ndim == 2:
                    labels, counts = np.unique(value, return_counts=True)
                    if len(labels) <= 100:
                        print("Label counts:", dict(zip(labels.tolist(), counts.tolist())))
                        print("Nonzero pixels:", int(np.count_nonzero(value)))
                if value.ndim == 3:
                    print("Nonfinite values:", int(value.size - np.count_nonzero(np.isfinite(value))))
        except Exception as exc:
            failures.append(path.name)
            print(f"ERROR: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError(f"Could not inspect: {failures}")


if __name__ == "__main__":
    main()
