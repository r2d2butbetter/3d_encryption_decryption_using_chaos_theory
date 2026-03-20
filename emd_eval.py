"""
Compute Earth Mover's Distance (EMD) between original OFF models
and the generated results (encrypted and/or decrypted) in output/.

Usage (PowerShell / cmd):
    python emd_eval.py                            # compare both encrypted & decrypted vs originals (Sinkhorn default)
  python emd_eval.py --subset decrypted         # only decrypted vs original
  python emd_eval.py --subset encrypted         # only encrypted vs original
  python emd_eval.py --class chair              # restrict to one class
  python emd_eval.py --max-points 1024          # cap points per cloud
    python emd_eval.py --no-normalize             # use raw coordinates
    python emd_eval.py --method hungarian         # exact matching (needs SciPy)
    python emd_eval.py --epsilon 0.05             # Sinkhorn regularization

Outputs CSV and JSON summaries and prints aggregates.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from off_io import read_off
from metrics import emd_point_cloud, sinkhorn_emd_point_cloud
from dataset_bootstrap import ensure_dataset_available

OUT_ROOT  = Path(__file__).parent / 'output'
DEFAULT_CLASS_HINTS = {
    'modelnet10': [
        'bathtub', 'bed', 'chair', 'desk', 'dresser',
        'monitor', 'night_stand', 'sofa', 'table', 'toilet',
    ],
    'modelnet40': None,
    'scanobjectnn': None,
}


essential_prefixes = ('encrypted_', 'decrypted_')


def _discover_classes(data_root: Path, class_hints: list[str] | None = None) -> list[str]:
    classes = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
    if class_hints:
        allow = set(class_hints)
        classes = [c for c in classes if c in allow]
    return classes


def map_output_to_original(cls: str, out_file: Path, data_root: Path) -> Optional[Path]:
    """
    Given an output filename like 'decrypted_chair_0123.off' inside output/cls,
    find the corresponding original OFF file under the selected dataset root.
    Try test/ then train/ splits.
    """
    stem = out_file.stem
    # Remove prefix 'encrypted_' or 'decrypted_'
    base = stem
    for p in essential_prefixes:
        if stem.startswith(p):
            base = stem[len(p):]
            break
    # Some outputs may include the class in the name already; keep as-is
    # Expect original like '<class>_0123.off'
    candidate = f"{base}.off"
    for split in ('test', 'train'):
        cand_path = data_root / cls / split / candidate
        if cand_path.exists():
            return cand_path

    # Alternate layout: <root>/<split>/<class>/<file>
    for split in ('test', 'train'):
        cand_path = data_root / split / cls / candidate
        if cand_path.exists():
            return cand_path

    # Flat class layout: <root>/<class>/<file>
    cand_path = data_root / cls / candidate
    if cand_path.exists():
        return cand_path

    # Fallback recursive lookup by basename
    found = list(data_root.rglob(candidate))
    if found:
        return found[0]
    return None


def gather_outputs(dataset: str, subset: str, only_class: Optional[str], data_root: Path, class_hints: list[str] | None) -> list[Tuple[str, Path, Path]]:
    """
    Return a list of tuples (subset, result_path, original_path) for evaluation.
    subset in {'encrypted','decrypted'}
    """
    tasks = []
    allowed_classes = set([only_class]) if only_class else set(_discover_classes(data_root, class_hints))

    # New layout support: output/<dataset>/<split>/<class>/*.off
    dataset_out_root = OUT_ROOT / dataset
    candidate_files: list[Path] = []
    if dataset_out_root.exists():
        candidate_files.extend(sorted(dataset_out_root.rglob('*.off')))

    # Legacy support: output/<class>/*.off
    for cls_dir in sorted([d for d in OUT_ROOT.iterdir() if d.is_dir()]):
        if cls_dir.name == dataset:
            continue
        candidate_files.extend(sorted(cls_dir.glob('*.off')))

    for out_file in candidate_files:
        cls = out_file.parent.name
        if allowed_classes and cls not in allowed_classes:
            continue

        name = out_file.name
        if subset == 'encrypted' and not name.startswith('encrypted_'):
            continue
        if subset == 'decrypted' and not name.startswith('decrypted_'):
            continue
        if subset == 'both' and not (name.startswith('encrypted_') or name.startswith('decrypted_')):
            continue

        orig = map_output_to_original(cls, out_file, data_root)
        if orig is None:
            continue
        tag = 'encrypted' if name.startswith('encrypted_') else 'decrypted'
        tasks.append((tag, out_file, orig))
    return tasks


def main():
    parser = argparse.ArgumentParser(description='Compute EMD between outputs and original OFF models')
    parser.add_argument('--dataset', choices=['modelnet10', 'modelnet40', 'scanobjectnn'], default='modelnet10',
                        help='Dataset preset used to resolve paths')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Optional explicit dataset root path')
    parser.add_argument('--no-download', action='store_true',
                        help='Do not auto-download missing dataset')
    parser.add_argument('--subset', choices=['encrypted','decrypted','both'], default='both',
                        help='Which outputs to compare against originals')
    parser.add_argument('--class', dest='only_class', type=str, default=None,
                        help='Restrict to a single class (e.g., chair)')
    parser.add_argument('--max-points', type=int, default=1024,
                        help='Maximum points per cloud (downsampled uniformly)')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Disable bbox normalization')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed for downsampling')
    parser.add_argument('--csv', type=str, default='emd_results.csv',
                        help='Output CSV path')
    parser.add_argument('--json', type=str, default='emd_results.json',
                        help='Output JSON path')
    parser.add_argument('--method', choices=['sinkhorn','hungarian','greedy'], default='sinkhorn',
                        help='EMD method (sinkhorn default; hungarian requires SciPy)')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Sinkhorn regularization strength (smaller -> closer to EMD, slower)')
    parser.add_argument('--max-iter', type=int, default=200,
                        help='Max Sinkhorn iterations')
    parser.add_argument('--tol', type=float, default=1e-3,
                        help='Sinkhorn marginal tolerance')
    args = parser.parse_args()

    subset = args.subset
    normalize = 'none' if args.no_normalize else 'bbox'
    data_root = ensure_dataset_available(
        args.dataset,
        args.data_root,
        auto_download=not args.no_download,
    )
    class_hints = DEFAULT_CLASS_HINTS.get(args.dataset)

    if data_root is None:
        print(f'Dataset root not found for {args.dataset}. Use --data-root to specify it.')
        return

    tasks = gather_outputs(args.dataset, subset, args.only_class, data_root, class_hints)
    if not tasks:
        print('No matching outputs found. Ensure output/<class> exists and dataset paths are correct.')
        return

    rows = []
    emd_by_tag = {'encrypted': [], 'decrypted': []}

    for tag, out_path, orig_path in tasks:
        # Read point sets (use vertices as point clouds)
        V_out, _ = read_off(str(out_path))
        V_org, _ = read_off(str(orig_path))

        if args.method == 'sinkhorn':
            emd = sinkhorn_emd_point_cloud(
                V_out, V_org,
                normalize=normalize,
                max_points=args.max_points,
                seed=args.seed,
                epsilon=args.epsilon,
                max_iter=args.max_iter,
                tol=args.tol,
            )
        elif args.method == 'hungarian':
            emd = emd_point_cloud(
                V_out, V_org,
                normalize=normalize,
                max_points=args.max_points,
                seed=args.seed,
                exact_if_scipy=True,
            )
        else:
            emd = emd_point_cloud(
                V_out, V_org,
                normalize=normalize,
                max_points=args.max_points,
                seed=args.seed,
                exact_if_scipy=False,
            )
        rows.append({
            'dataset': args.dataset,
            'class': out_path.parent.name,
            'subset': tag,
            'output_file': str(out_path),
            'original_file': str(orig_path),
            'n_out': len(V_out),
            'n_org': len(V_org),
            'emd': emd,
        })
        emd_by_tag[tag].append(emd)
        print(f"[{tag:9}] {out_path.parent.name:12} {out_path.name:32}  EMD={emd:.6f}")

    # Write CSV
    csv_path = Path(args.csv)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV: {csv_path}")

    summary = {
        'dataset': args.dataset,
        'data_root': str(data_root),
        'subset': subset,
        'method': args.method,
        'normalize': normalize,
        'generated_at_unix': int(time.time()),
        'count': len(rows),
        'encrypted_mean_emd': float(np.mean(emd_by_tag['encrypted'])) if emd_by_tag['encrypted'] else None,
        'decrypted_mean_emd': float(np.mean(emd_by_tag['decrypted'])) if emd_by_tag['decrypted'] else None,
    }

    json_path = Path(args.json)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'rows': rows}, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Summary
    for tag in ('encrypted', 'decrypted'):
        vals = emd_by_tag[tag]
        if vals:
            print(f"{tag.capitalize():9}  mean EMD = {np.mean(vals):.6f}  (n={len(vals)})")


if __name__ == '__main__':
    main()
