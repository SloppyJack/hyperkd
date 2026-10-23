"""Run saved HyperKD weights and export classification maps. No training."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import cohen_kappa_score
import torch

from datasets.preprocess import applyPCA, my_normalize, mirror_hsi
from networks.LSGA_ViT import LSGAVIT
from networks.network import LLL_Net


NAMES = ['Pavia', 'IndianPine', 'Salinas', 'Houston']
CLASSES = [9, 16, 16, 15]
OFFSETS = [0, 9, 25, 41]


def predict(model, padded, points, batch_size, device):
    result = np.empty(len(points), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(points), batch_size):
            coords = points[start:start + batch_size]
            patches = np.stack([padded[r:r + 7, c:c + 7, :] for r, c in coords])
            x = torch.from_numpy(np.ascontiguousarray(patches.transpose(0, 3, 1, 2), dtype=np.float32))
            outputs = model(x.to(device))
            result[start:start + len(coords)] = torch.cat(outputs, dim=1).argmax(1).cpu().numpy()
            if start == 0 or start + batch_size >= len(points) or start // batch_size % 100 == 0:
                print(f'  Predicted {min(start + batch_size, len(points))}/{len(points)} pixels', flush=True)
    return result


def save_maps(out, name, stage, gt, te, global_map, classes, offset, metrics):
    # 0=unlabeled, 1..C=local classes, C+1=predicted a class from another task.
    colors = np.vstack(([0, 0, 0, 1], plt.get_cmap('tab20')(np.arange(classes)), [1, 0, 1, 1]))
    display = np.zeros(gt.shape, dtype=np.uint8)
    foreground = gt > 0
    local = global_map - offset + 1
    display[foreground] = np.where((local[foreground] >= 1) & (local[foreground] <= classes),
                                   local[foreground], classes + 1)
    test_pred = np.where(te > 0, display, 0)
    correct = (te > 0) & (global_map == te.astype(np.int64) - 1 + offset)
    errors = np.zeros((*gt.shape, 3), dtype=np.uint8)
    errors[correct] = [40, 190, 80]
    errors[(te > 0) & ~correct] = [230, 45, 45]
    for suffix, rgb in [('ground_truth', colors[gt.astype(int)]),
                        ('prediction_labeled', colors[display]),
                        ('prediction_test', colors[test_pred]), ('test_errors', errors)]:
        plt.imsave(out / f'{name}_{suffix}.png', rgb)
    fig, axes = plt.subplots(1, 4, figsize=(16, 7))
    panels = [(colors[gt.astype(int)], 'Ground truth (labeled pixels)'),
              (colors[display], 'Prediction (includes training pixels)'),
              (colors[test_pred], f'Test prediction | OA={metrics["OA"] * 100:.2f}%'),
              (errors, 'Test errors: red=wrong, green=correct')]
    for ax, (rgb, title) in zip(axes, panels):
        ax.imshow(rgb, interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    legend = [Patch(facecolor=colors[c], label=f'Class {c}') for c in range(1, classes + 1)]
    legend += [Patch(facecolor='magenta', label='Other-task prediction')]
    fig.legend(handles=legend, loc='lower center', ncol=6, fontsize=8)
    fig.suptitle(f'{name} | checkpoint task_{stage}.pth | task-agnostic classification')
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    fig.savefig(out / f'{name}_comparison.png', dpi=180)
    plt.close(fig)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=range(4), default=3,
                        help='Checkpoint index: 0=after PU, 3=after all four tasks')
    parser.add_argument('--seed', type=int, default=0, help='Must match training seed for PCA reconstruction')
    parser.add_argument('--data-dir', type=Path, default=base / 'data' / 'prepared')
    parser.add_argument('--output-dir', type=Path, default=base / 'output' / 'classification_maps')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    checkpoint = base / 'model_path' / f'task_{args.stage}.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Upload/use the server checkpoint: {checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LLL_Net(LSGAVIT(img_size=7, patch_size=3, in_chans=36,
                           num_classes=15, embed_dim=120, depths=[2],
                           num_heads=[12, 12, 12, 24]))
    for count in CLASSES[:args.stage + 1]:
        model.add_head(count)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f'Loaded {checkpoint} on {device}', flush=True)
    out = args.output_dir / f'task_{args.stage}'
    out.mkdir(parents=True, exist_ok=True)
    summary = {'checkpoint': str(checkpoint), 'seed': args.seed, 'metrics_scope': 'TE only',
               'prediction_scope': 'TR+TE labeled pixels; background excluded', 'datasets': {}}
    # Training calls PCA in dataset order using the global NumPy RNG. Reproduce it.
    np.random.seed(args.seed)
    for i, name in enumerate(NAMES[:args.stage + 1]):
        print(f'Processing {name}', flush=True)
        raw = loadmat(args.data_dir / f'{name}.mat')
        tr, te = raw['TR'], raw['TE']
        if np.any((tr > 0) & (te > 0)):
            raise ValueError(f'{name}: train/test masks overlap')
        gt = tr + te
        cube = my_normalize(applyPCA(raw['input'], 36))
        if cube.shape[:2] != gt.shape or not np.isfinite(cube).all():
            raise ValueError(f'{name}: invalid preprocessing output')
        padded = mirror_hsi(*cube.shape, cube, patch=7)
        points = np.argwhere(gt > 0)
        predictions = predict(model, padded, points, args.batch_size, device)
        global_map = np.full(gt.shape, -1, dtype=np.int16)
        global_map[points[:, 0], points[:, 1]] = predictions
        true = te[te > 0].astype(np.int64) - 1 + OFFSETS[i]
        pred = global_map[te > 0]
        per_class = [float((pred[true == c] == c).mean())
                     for c in range(OFFSETS[i], OFFSETS[i] + CLASSES[i])]
        metrics = {'OA': float((true == pred).mean()), 'AA': float(np.mean(per_class)),
                   'Kappa': float(cohen_kappa_score(true, pred)),
                   'per_class_accuracy': per_class, 'test_pixels': int(len(true))}
        save_maps(out, name, args.stage, gt, te, global_map, CLASSES[i], OFFSETS[i], metrics)
        np.savez_compressed(out / f'{name}_predictions.npz', prediction_global=global_map,
                            TR=tr, TE=te)
        summary['datasets'][name] = metrics
        print(f'{name}: OA={metrics["OA"]:.4%}, AA={metrics["AA"]:.4%}, Kappa={metrics["Kappa"]:.4f}', flush=True)
        (out / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Done. Images and metrics saved to {out}', flush=True)


if __name__ == '__main__':
    main()
