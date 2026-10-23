"""Create reproducible proxy splits, NOT the unavailable author splits.

Uses Table I training quotas and all remaining labeled pixels as test data.
Retains original spatial orientation and original (uncorrected) spectral bands.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


SPECS = [
    ("Pavia", "PaviaU.mat", "paviaU", "PaviaU_gt.mat", "paviaU_gt",
     [548, 540, 392, 524, 265, 532, 375, 514, 231]),
    ("IndianPine", "Indian_pines.mat", "indian_pines", "Indian_pines_gt.mat", "indian_pines_gt",
     [15, 50, 50, 50, 50, 50, 15, 50, 15, 50, 50, 50, 50, 50, 50, 50]),
    ("Salinas", "Salinas.mat", "salinas", "Salinas_gt.mat", "salinas_gt", [200] * 16),
    ("Houston", "Houstondata.mat", "Houstondata", "Houstonlabel.mat", "Houstonlabel",
     [198, 190, 192, 188, 186, 182, 196, 191, 193, 191, 181, 192, 184, 181, 187]),
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=base / "data")
    parser.add_argument("--output-dir", type=Path, default=base / "data" / "prepared")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    targets = [args.output_dir / (s[0] + ".mat") for s in SPECS]
    targets.append(args.output_dir / "split_manifest.json")
    if any(p.exists() for p in targets):
        raise FileExistsError("Output already exists; choose a new --output-dir to preserve existing splits")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": "proxy random split using Table I training quotas; not author split",
                "seed": args.seed, "numpy_version": np.__version__, "datasets": []}
    for name, image_file, image_key, label_file, label_key, quotas in SPECS:
        image_path, label_path = args.raw_dir / image_file, args.raw_dir / label_file
        image = loadmat(image_path)[image_key]
        labels = loadmat(label_path)[label_key]
        if image.ndim != 3 or labels.ndim != 2 or image.shape[:2] != labels.shape:
            raise ValueError(f"{name}: incompatible shapes {image.shape}, {labels.shape}")
        if not np.isfinite(image).all():
            raise ValueError(f"{name}: image contains nonfinite values")
        if not np.array_equal(np.unique(labels), np.arange(len(quotas) + 1)):
            raise ValueError(f"{name}: expected background 0 and consecutive class labels")
        rng = np.random.default_rng(args.seed)
        train = np.zeros_like(labels)
        test = labels.copy()
        rows = []
        for cls, quota in enumerate(quotas, 1):
            candidates = np.flatnonzero(labels == cls)
            if len(candidates) <= quota:
                raise ValueError(f"{name}: class {cls} has too few pixels for train/test")
            selected = rng.choice(candidates, quota, replace=False)
            train.flat[selected] = cls
            test.flat[selected] = 0
            rows.append({"class": cls, "train": quota, "test": int(len(candidates) - quota)})
        assert not np.any((train > 0) & (test > 0))
        assert np.array_equal(train + test, labels)
        destination = args.output_dir / f"{name}.mat"
        savemat(destination, {"input": image, "TR": train, "TE": test}, do_compression=True)
        # Verify the actual saved masks as well as the in-memory split.
        saved = loadmat(destination, variable_names=["TR", "TE"])
        assert np.array_equal(saved["TR"], train) and np.array_equal(saved["TE"], test)
        record = {"name": name, "shape": list(image.shape), "classes": rows,
                  "sources": {image_file: digest(image_path), label_file: digest(label_path)}}
        manifest["datasets"].append(record)
        print(f"{name}: shape={image.shape}, train={np.count_nonzero(train)}, "
              f"test={np.count_nonzero(test)} -> {destination}", flush=True)
    (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Prepared. These are proxy splits, not the original paper splits.")


if __name__ == "__main__":
    main()
