"""
Demo: 3D model encryption on OFF datasets.

Usage:
    python demo.py                                        # one model per class
    python demo.py --model path/to/file.off               # single file
    python demo.py --all --dataset modelnet40 --split test
    python demo.py --all --dataset scanobjectnn --split all

Outputs are saved under output/<dataset>/...
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from off_io        import read_off, write_off
from chaos_encrypt import encrypt, decrypt
from metrics       import security_report, print_report, npcr_uaci_paper, sinkhorn_emd_point_cloud
from dataset_bootstrap import ensure_dataset_available


# ---------------------------------------------------------------------------
# Default encryption key (user-configurable)
# ---------------------------------------------------------------------------
DEFAULT_KEY = {
    'u1'  : 0.37891,   # logistic map seed (0 < u1 < 1)
    's1'  : 0.45632,   # tent map initial value (0 < s1 < 1)
    'beta': 1.99964,   # tent map control parameter (1 < beta < 2)
    'L'   : 100,       # logistic transient
    'T'   : 100,       # LDCML transient
}

DEFAULT_CLASS_HINTS = {
    'modelnet10': [
        'bathtub', 'bed', 'chair', 'desk', 'dresser',
        'monitor', 'night_stand', 'sofa', 'table', 'toilet',
    ],
    'modelnet40': None,
    'scanobjectnn': None,
}


def _subdirs(root: Path) -> list[Path]:
    return sorted([d for d in root.iterdir() if d.is_dir()], key=lambda d: d.name)


def _filter_classes(classes: list[str], class_hints: list[str] | None) -> list[str]:
    if not class_hints:
        return classes
    allowed = set(class_hints)
    return [c for c in classes if c in allowed]


def find_off_files(root: Path, split: str = 'test', class_hints: list[str] | None = None) -> dict[str, list[Path]]:
    """Return class->OFF file list for common dataset layouts."""
    files: dict[str, list[Path]] = {}
    split_values = ('train', 'test') if split == 'both' else (split,)

    # Layout A: <root>/<class>/<split>/*.off
    class_dirs = _subdirs(root)
    class_names = _filter_classes([d.name for d in class_dirs], class_hints)
    for cls in class_names:
        cls_dir = root / cls
        matches: list[Path] = []
        for sp in split_values:
            if sp == 'all':
                matches.extend(sorted(cls_dir.rglob('*.off')))
            else:
                sp_dir = cls_dir / sp
                if sp_dir.exists():
                    matches.extend(sorted(sp_dir.glob('*.off')))
        if matches:
            files[cls] = matches
    if files:
        return files

    # Layout B: <root>/<split>/<class>/*.off
    if split != 'all':
        files = {}
        for sp in split_values:
            sp_root = root / sp
            if not sp_root.exists() or not sp_root.is_dir():
                continue
            for cls_dir in _subdirs(sp_root):
                cls = cls_dir.name
                if class_hints and cls not in class_hints:
                    continue
                matches = sorted(cls_dir.glob('*.off'))
                if matches:
                    files.setdefault(cls, []).extend(matches)
        if files:
            return files

    # Layout C: <root>/<class>/*.off
    files = {}
    for cls_dir in class_dirs:
        cls = cls_dir.name
        if class_hints and cls not in class_hints:
            continue
        matches = sorted(cls_dir.glob('*.off'))
        if matches:
            files[cls] = matches
    if files:
        return files

    # Layout D fallback: recursively collect all OFF files
    files = {}
    for off in sorted(root.rglob('*.off')):
        cls = off.parent.name
        if class_hints and cls not in class_hints:
            continue
        files.setdefault(cls, []).append(off)
    return files


# ---------------------------------------------------------------------------
# Single-file pipeline
# ---------------------------------------------------------------------------

def process_file(
    off_path: Path,
    out_dir: Path,
    key: dict,
    verbose: bool = True,
    compute_emd: bool = True,
    emd_normalize: str = 'bbox',
    emd_max_points: int = 1024,
    emd_epsilon: float = 0.05,
    emd_max_iter: int = 200,
    emd_tol: float = 1e-3,
):
    """
    Encrypt and decrypt a single OFF file.  Save outputs and report metrics.

    Returns the security report dict.
    """
    # ── Read ─────────────────────────────────────────────────────────────────
    vertices, faces = read_off(str(off_path))
    s = len(vertices)
    if verbose:
        print(f"\n{'─'*55}")
        print(f"File     : {off_path.name}")
        print(f"Vertices : {s}   Faces : {len(faces)}")

    # ── Encrypt ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    enc_verts, enc_key = encrypt(vertices, key)
    t_enc = time.perf_counter() - t0

    # ── Decrypt ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    dec_verts = decrypt(enc_verts, enc_key)
    t_dec = time.perf_counter() - t0

    if verbose:
        print(f"Encrypt  : {t_enc*1000:.1f} ms   Decrypt : {t_dec*1000:.1f} ms")

    # ── Save outputs ─────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = off_path.stem

    enc_path = out_dir / f"encrypted_{stem}.off"
    dec_path = out_dir / f"decrypted_{stem}.off"
    key_path = out_dir / f"key_{stem}.json"

    write_off(str(enc_path), enc_verts, faces)
    write_off(str(dec_path), dec_verts, faces)

    # Save enc_key as JSON (convert numpy types for JSON serialisation)
    key_serial = {k: (int(v) if isinstance(v, (np.integer,))
                      else float(v) if isinstance(v, (np.floating, float))
                      else v)
                  for k, v in enc_key.items()}
    with open(key_path, 'w') as fk:
        json.dump(key_serial, fk, indent=2)

    # ── Security metrics ─────────────────────────────────────────────────────
    report = security_report(vertices, enc_verts, dec_verts)

    # Paper Section 5.7 differential attack:
    # increment the first plaintext entry by +1, encrypt with same key,
    # then compute NPCR/UACI between the two ciphertexts.
    vertices_mod = vertices.copy()
    vertices_mod.reshape(-1)[0] += 1.0
    enc_verts_mod, _ = encrypt(vertices_mod, key)
    report['npcr_diff'], report['uaci_diff'] = npcr_uaci_paper(
        enc_verts,
        enc_verts_mod,
        intensity_denominator=float(enc_key['bitxor_1'])
    )

    report['file']      = str(off_path)
    report['n_verts']   = s
    report['t_enc_ms']  = round(t_enc * 1000, 2)
    report['t_dec_ms']  = round(t_dec * 1000, 2)

    if compute_emd:
        t0 = time.perf_counter()
        report['emd_encrypted'] = sinkhorn_emd_point_cloud(
            enc_verts,
            vertices,
            normalize=emd_normalize,
            max_points=emd_max_points,
            seed=0,
            epsilon=emd_epsilon,
            max_iter=emd_max_iter,
            tol=emd_tol,
        )
        report['emd_decrypted'] = sinkhorn_emd_point_cloud(
            dec_verts,
            vertices,
            normalize=emd_normalize,
            max_points=emd_max_points,
            seed=0,
            epsilon=emd_epsilon,
            max_iter=emd_max_iter,
            tol=emd_tol,
        )
        report['t_emd_ms'] = round((time.perf_counter() - t0) * 1000, 2)
    else:
        report['emd_encrypted'] = None
        report['emd_decrypted'] = None
        report['t_emd_ms'] = 0.0

    if verbose:
        print_report(report)

    return report


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

OUT_ROOT = Path(__file__).parent / 'output'


def run_one_per_class(
    key: dict,
    dataset: str,
    data_root: Path,
    class_hints: list[str] | None,
    emd_cfg: dict,
):
    """Encrypt/decrypt the first test file of each class and show metrics."""
    all_files = find_off_files(data_root, split='test', class_hints=class_hints)
    if not all_files:
        all_files = find_off_files(data_root, split='all', class_hints=class_hints)

    all_reports = []

    for cls, file_list in all_files.items():
        out_dir = OUT_ROOT / dataset / 'sample' / cls
        report = process_file(file_list[0], out_dir, key, **emd_cfg)
        report['class'] = cls
        report['dataset'] = dataset
        report['split'] = 'sample'
        all_reports.append(report)

    _print_summary(all_reports)
    return all_reports


def run_all(
    key: dict,
    dataset: str,
    data_root: Path,
    class_hints: list[str] | None,
    split: str = 'test',
    print_summary: bool = True,
    emd_cfg: dict | None = None,
):
    """Encrypt/decrypt every file in the given split."""
    emd_cfg = emd_cfg or {}
    all_files = find_off_files(data_root, split=split, class_hints=class_hints)
    total = sum(len(v) for v in all_files.values())
    done  = 0
    all_reports = []

    for cls, file_list in all_files.items():
        out_dir = OUT_ROOT / dataset / split / cls
        for fpath in file_list:
            done += 1
            print(f"[{done}/{total}] {cls} / {fpath.name}", end='  ', flush=True)
            report = process_file(fpath, out_dir, key, verbose=False, **emd_cfg)
            report['class'] = cls
            report['dataset'] = dataset
            report['split'] = split
            all_reports.append(report)
            print(f"enc={report['t_enc_ms']:.0f}ms  "
                  f"dec={report['t_dec_ms']:.0f}ms  "
                  f"H_c={report['entropy_cipher']:.4f}  "
                  f"NPCR={report['npcr_diff']:.1f}%  "
                  f"EMD_d={report['emd_decrypted']:.4f}" if report['emd_decrypted'] is not None else "")
    if print_summary:
        _print_summary(all_reports)
    return all_reports


def run_all_both(
    key: dict,
    dataset: str,
    data_root: Path,
    class_hints: list[str] | None,
    emd_cfg: dict,
):
    """Encrypt/decrypt every file across both train and test splits."""
    combined = []
    for split in ('train', 'test'):
        print(f"\nProcessing split: {split}\n" + "-"*40)
        reports = run_all(
            key,
            dataset=dataset,
            data_root=data_root,
            class_hints=class_hints,
            split=split,
            print_summary=False,
            emd_cfg=emd_cfg,
        )
        combined.extend(reports)

    if not combined:
        print("No train/test split found. Falling back to split=all.")
        combined = run_all(
            key,
            dataset=dataset,
            data_root=data_root,
            class_hints=class_hints,
            split='all',
            print_summary=False,
            emd_cfg=emd_cfg,
        )

    _print_summary(combined)
    return combined


def _print_summary(reports: list):
    """Print aggregate statistics across multiple files."""
    if not reports:
        return
    have_emd = any(r.get('emd_decrypted') is not None for r in reports)
    width = 92 if have_emd else 65
    print(f"\n{'═'*width}")
    print(f"  SUMMARY  ({len(reports)} files)")
    print(f"{'═'*width}")
    header = f"  {'Class':<14} {'H_plain':>8} {'H_cipher':>9} {'NPCR%':>7} {'UACI%':>7} {'Max_err':>10}"
    if have_emd:
        header += f" {'EMD_enc':>9} {'EMD_dec':>9}"
    print(header)
    print(f"  {'─'*(width-3)}")
    for r in reports:
        cls   = r.get('class', 'N/A')
        row = (f"  {cls:<14} "
              f"{r['entropy_plain']:>8.4f} "
              f"{r['entropy_cipher']:>9.4f} "
              f"{r['npcr_diff']:>7.2f} "
              f"{r['uaci_diff']:>7.2f} "
               f"{r['max_reconstruction_error']:>10.2e}")
        if have_emd:
            emd_enc = r['emd_encrypted'] if r['emd_encrypted'] is not None else float('nan')
            emd_dec = r['emd_decrypted'] if r['emd_decrypted'] is not None else float('nan')
            row += f" {emd_enc:>9.4f} {emd_dec:>9.4f}"
        print(row)
    # Averages
    avg_H   = np.mean([r['entropy_cipher'] for r in reports])
    avg_npcr= np.mean([r['npcr_diff'] for r in reports])
    avg_uaci= np.mean([r['uaci_diff'] for r in reports])
    avg_err = np.mean([r['max_reconstruction_error'] for r in reports])
    avg_row = f"  {'AVERAGE':<14} {'':>8} {avg_H:>9.4f} {avg_npcr:>7.2f} {avg_uaci:>7.2f} {avg_err:>10.2e}"
    if have_emd:
        emd_enc_vals = [r['emd_encrypted'] for r in reports if r['emd_encrypted'] is not None]
        emd_dec_vals = [r['emd_decrypted'] for r in reports if r['emd_decrypted'] is not None]
        avg_emd_enc = float(np.mean(emd_enc_vals)) if emd_enc_vals else float('nan')
        avg_emd_dec = float(np.mean(emd_dec_vals)) if emd_dec_vals else float('nan')
        avg_row += f" {avg_emd_enc:>9.4f} {avg_emd_dec:>9.4f}"
    print(f"  {'─'*(width-3)}")
    print(avg_row)
    print(f"{'═'*width}\n")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _build_aggregate(reports: list[dict]) -> dict:
    if not reports:
        return {}
    out = {
        'count': len(reports),
        'entropy_cipher_mean': float(np.mean([r['entropy_cipher'] for r in reports])),
        'npcr_mean': float(np.mean([r['npcr_diff'] for r in reports])),
        'uaci_mean': float(np.mean([r['uaci_diff'] for r in reports])),
        'reconstruction_error_mean': float(np.mean([r['max_reconstruction_error'] for r in reports])),
        'enc_time_ms_mean': float(np.mean([r['t_enc_ms'] for r in reports])),
        'dec_time_ms_mean': float(np.mean([r['t_dec_ms'] for r in reports])),
    }
    emd_enc_vals = [r['emd_encrypted'] for r in reports if r.get('emd_encrypted') is not None]
    emd_dec_vals = [r['emd_decrypted'] for r in reports if r.get('emd_decrypted') is not None]
    if emd_enc_vals:
        out['emd_encrypted_mean'] = float(np.mean(emd_enc_vals))
    if emd_dec_vals:
        out['emd_decrypted_mean'] = float(np.mean(emd_dec_vals))
    return out


def save_metrics_json(path: Path, dataset: str, data_root: Path, split: str, reports: list[dict], key: dict, emd_cfg: dict):
    payload = {
        'dataset': dataset,
        'data_root': str(data_root),
        'split': split,
        'generated_at_unix': int(time.time()),
        'key': key,
        'emd_config': emd_cfg,
        'aggregate': _build_aggregate(reports),
        'reports': reports,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(payload), f, indent=2)
    print(f"Saved metrics JSON: {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='3D chaos-based encryption demo on OFF datasets')
    parser.add_argument('--model',   type=str, default=None,
                        help='Path to a single .off file to process')
    parser.add_argument('--all',     action='store_true',
                        help='Process all files for the selected split(s)')
    parser.add_argument('--dataset', type=str, default='modelnet10',
                        choices=['modelnet10', 'modelnet40', 'scanobjectnn'],
                        help='Dataset preset to load from data/')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Optional explicit dataset root path')
    parser.add_argument('--no-download', action='store_true',
                        help='Do not auto-download missing dataset')
    parser.add_argument('--split',   type=str, default='test',
                        choices=['train', 'test', 'both', 'all'],
                        help='Dataset split to use')
    parser.add_argument('--u1',      type=float, default=DEFAULT_KEY['u1'])
    parser.add_argument('--s1',      type=float, default=DEFAULT_KEY['s1'])
    parser.add_argument('--beta',    type=float, default=DEFAULT_KEY['beta'])
    parser.add_argument('--no-emd', action='store_true',
                        help='Disable Sinkhorn EMD computation')
    parser.add_argument('--emd-max-points', type=int, default=1024,
                        help='Max points per cloud for EMD/Sinkhorn')
    parser.add_argument('--emd-epsilon', type=float, default=0.05,
                        help='Sinkhorn regularization strength')
    parser.add_argument('--emd-max-iter', type=int, default=200,
                        help='Sinkhorn max iterations')
    parser.add_argument('--emd-tol', type=float, default=1e-3,
                        help='Sinkhorn convergence tolerance')
    parser.add_argument('--emd-no-normalize', action='store_true',
                        help='Disable bbox normalization for EMD')
    parser.add_argument('--metrics-json', type=str, default=None,
                        help='Output JSON path for run metrics')
    args = parser.parse_args()

    key = {**DEFAULT_KEY, 'u1': args.u1, 's1': args.s1, 'beta': args.beta}
    data_root = ensure_dataset_available(
        args.dataset,
        args.data_root,
        auto_download=not args.no_download,
    )
    class_hints = DEFAULT_CLASS_HINTS.get(args.dataset)

    if data_root is None and not args.model:
        print(f"Dataset root not found for {args.dataset}. Use --data-root to specify it.")
        return

    emd_cfg = {
        'compute_emd': not args.no_emd,
        'emd_normalize': 'none' if args.emd_no_normalize else 'bbox',
        'emd_max_points': args.emd_max_points,
        'emd_epsilon': args.emd_epsilon,
        'emd_max_iter': args.emd_max_iter,
        'emd_tol': args.emd_tol,
    }

    reports: list[dict] = []
    run_label = args.split if args.all else 'sample'

    if args.model:
        off_path = Path(args.model)
        out_dir = OUT_ROOT / args.dataset / 'single' / off_path.parent.name
        report = process_file(off_path, out_dir, key, **emd_cfg)
        report['class'] = off_path.parent.name
        report['dataset'] = args.dataset
        report['split'] = 'single'
        reports = [report]
    elif args.all:
        if args.split == 'both':
            reports = run_all_both(key, args.dataset, data_root, class_hints, emd_cfg)
        else:
            reports = run_all(
                key,
                dataset=args.dataset,
                data_root=data_root,
                class_hints=class_hints,
                split=args.split,
                emd_cfg=emd_cfg,
            )
    else:
        print("Running one model per class …\n")
        reports = run_one_per_class(key, args.dataset, data_root, class_hints, emd_cfg)

    if reports:
        if args.metrics_json:
            json_path = Path(args.metrics_json)
        else:
            stamp = time.strftime('%Y%m%d_%H%M%S')
            json_path = Path('output') / f"metrics_{args.dataset}_{run_label}_{stamp}.json"
        save_metrics_json(json_path, args.dataset, data_root if data_root else Path('.'), args.split, reports, key, emd_cfg)


if __name__ == '__main__':
    main()
