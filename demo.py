"""
Demo: 3D model encryption on ModelNet10 OFF files.

Usage:
    python demo.py                         # encrypts one model per class (test split)
    python demo.py --model path/to/file.off
    python demo.py --all                   # process all test files

Outputs are saved to:
    output/<class>/encrypted_<name>.off
    output/<class>/decrypted_<name>.off
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from off_io        import read_off, write_off
from chaos_encrypt import encrypt, decrypt
from metrics       import security_report, print_report, npcr_uaci_paper


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


# ---------------------------------------------------------------------------
# Single-file pipeline
# ---------------------------------------------------------------------------

def process_file(off_path: Path, out_dir: Path, key: dict, verbose: bool = True):
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

    if verbose:
        print_report(report)

    return report


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

MODELNET10_CLASSES = [
    'bathtub', 'bed', 'chair', 'desk', 'dresser',
    'monitor', 'night_stand', 'sofa', 'table', 'toilet',
]

DATA_ROOT = Path(__file__).parent / 'data' / 'ModelNet10' / 'ModelNet10'
OUT_ROOT  = Path(__file__).parent / 'output'


def find_off_files(split: str = 'test'):
    """Return a dict mapping class_name → list of OFF file paths."""
    files = {}
    for cls in MODELNET10_CLASSES:
        cls_dir = DATA_ROOT / cls / split
        off_list = sorted(cls_dir.glob('*.off')) if cls_dir.exists() else []
        if off_list:
            files[cls] = off_list
    return files


def run_one_per_class(key: dict):
    """Encrypt/decrypt the first test file of each class and show metrics."""
    all_files = find_off_files('test')
    all_reports = []

    for cls, file_list in all_files.items():
        out_dir = OUT_ROOT / cls
        report  = process_file(file_list[0], out_dir, key)
        report['class'] = cls
        all_reports.append(report)

    _print_summary(all_reports)
    return all_reports


def run_all(key: dict, split: str = 'test'):
    """Encrypt/decrypt every file in the given split."""
    all_files = find_off_files(split)
    total = sum(len(v) for v in all_files.values())
    done  = 0
    all_reports = []

    for cls, file_list in all_files.items():
        out_dir = OUT_ROOT / cls
        for fpath in file_list:
            done += 1
            print(f"[{done}/{total}] {cls} / {fpath.name}", end='  ', flush=True)
            report = process_file(fpath, out_dir, key, verbose=False)
            report['class'] = cls
            all_reports.append(report)
            print(f"enc={report['t_enc_ms']:.0f}ms  "
                  f"dec={report['t_dec_ms']:.0f}ms  "
                  f"H_c={report['entropy_cipher']:.4f}  "
                f"NPCR={report['npcr_diff']:.1f}%")

    _print_summary(all_reports)
    return all_reports


def _print_summary(reports: list):
    """Print aggregate statistics across multiple files."""
    if not reports:
        return
    print(f"\n{'═'*65}")
    print(f"  SUMMARY  ({len(reports)} files)")
    print(f"{'═'*65}")
    header = f"  {'Class':<14} {'H_plain':>8} {'H_cipher':>9} {'NPCR%':>7} {'UACI%':>7} {'Max_err':>10}"
    print(header)
    print(f"  {'─'*62}")
    for r in reports:
        cls   = r.get('class', 'N/A')
        fname = Path(r['file']).name
        print(f"  {cls:<14} "
              f"{r['entropy_plain']:>8.4f} "
              f"{r['entropy_cipher']:>9.4f} "
              f"{r['npcr_diff']:>7.2f} "
              f"{r['uaci_diff']:>7.2f} "
              f"{r['max_reconstruction_error']:>10.2e}")
    # Averages
    avg_H   = np.mean([r['entropy_cipher'] for r in reports])
    avg_npcr= np.mean([r['npcr_diff'] for r in reports])
    avg_uaci= np.mean([r['uaci_diff'] for r in reports])
    avg_err = np.mean([r['max_reconstruction_error'] for r in reports])
    print(f"  {'─'*62}")
    print(f"  {'AVERAGE':<14} {'':>8} {avg_H:>9.4f} {avg_npcr:>7.2f} {avg_uaci:>7.2f} {avg_err:>10.2e}")
    print(f"{'═'*65}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='3D chaos-based encryption demo on ModelNet10')
    parser.add_argument('--model',   type=str, default=None,
                        help='Path to a single .off file to process')
    parser.add_argument('--all',     action='store_true',
                        help='Process ALL test files (slow)')
    parser.add_argument('--split',   type=str, default='test',
                        choices=['train', 'test'],
                        help='Dataset split to use (default: test)')
    parser.add_argument('--u1',      type=float, default=DEFAULT_KEY['u1'])
    parser.add_argument('--s1',      type=float, default=DEFAULT_KEY['s1'])
    parser.add_argument('--beta',    type=float, default=DEFAULT_KEY['beta'])
    args = parser.parse_args()

    key = {**DEFAULT_KEY, 'u1': args.u1, 's1': args.s1, 'beta': args.beta}

    if args.model:
        off_path = Path(args.model)
        out_dir  = OUT_ROOT / off_path.parent.name
        process_file(off_path, out_dir, key)
    elif args.all:
        run_all(key, split=args.split)
    else:
        print("Running one model per class (test split) …\n")
        run_one_per_class(key)


if __name__ == '__main__':
    main()
