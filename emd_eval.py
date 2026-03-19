"""
Compute Earth Mover's Distance (EMD) between original ModelNet10 OFF models
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

Outputs a CSV summary emd_results.csv in the repo root and prints aggregates.
"""

import argparse
import csv
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from off_io import read_off
from metrics import emd_point_cloud, sinkhorn_emd_point_cloud

DATA_ROOT = Path(__file__).parent / 'data' / 'ModelNet10' / 'ModelNet10'
OUT_ROOT  = Path(__file__).parent / 'output'
MODELNET10_CLASSES = [
    'bathtub', 'bed', 'chair', 'desk', 'dresser',
    'monitor', 'night_stand', 'sofa', 'table', 'toilet',
]


essential_prefixes = ('encrypted_', 'decrypted_')

def map_output_to_original(cls: str, out_file: Path) -> Optional[Path]:
    """
    Given an output filename like 'decrypted_chair_0123.off' inside output/cls,
    find the corresponding original OFF file under data/ModelNet10/ModelNet10.
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
        cand_path = DATA_ROOT / cls / split / candidate
        if cand_path.exists():
            return cand_path
    return None


def gather_outputs(subset: str, only_class: Optional[str]) -> list[Tuple[str, Path, Path]]:
    """
    Return a list of tuples (subset, result_path, original_path) for evaluation.
    subset in {'encrypted','decrypted'}
    """
    tasks = []
    classes = [only_class] if only_class else MODELNET10_CLASSES
    for cls in classes:
        out_dir = OUT_ROOT / cls
        if not out_dir.exists():
            continue
        for out_file in sorted(out_dir.glob('*.off')):
            name = out_file.name
            if subset == 'encrypted' and not name.startswith('encrypted_'):
                continue
            if subset == 'decrypted' and not name.startswith('decrypted_'):
                continue
            if subset == 'both' and not (name.startswith('encrypted_') or name.startswith('decrypted_')):
                continue
            orig = map_output_to_original(cls, out_file)
            if orig is None:
                continue
            tag = 'encrypted' if name.startswith('encrypted_') else 'decrypted'
            tasks.append((tag, out_file, orig))
    return tasks


def main():
    parser = argparse.ArgumentParser(description='Compute EMD between outputs and original OFF models')
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

    tasks = gather_outputs(subset, args.only_class)
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

    # Summary
    for tag in ('encrypted', 'decrypted'):
        vals = emd_by_tag[tag]
        if vals:
            print(f"{tag.capitalize():9}  mean EMD = {np.mean(vals):.6f}  (n={len(vals)})")


if __name__ == '__main__':
    main()
