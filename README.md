# Chaos-Based 3D Model Encryption (ModelNet10 OFF)

An implementation of a chaos-theory-based encryption/decryption pipeline for 3D models in OFF format, adapted from the paper “A novel 3-D image encryption algorithm based on SHA-256 and chaos theory” (Singh et al., Alexandria Engineering Journal, 2025). This repo demonstrates encrypting and decrypting ModelNet10 meshes and evaluates security metrics commonly used in the literature.

## Overview
- Algorithm stages:
  - Logistic map: scramble vertex coordinates and insert random points.
  - LDCML lattice: reconfusion + diffusion via XOR-chain and fractional scrambling.
  - Tent map: final confusion permutation.
- Plaintext-derived parameters: A SHA-256 digest of the plaintext vertices seeds several chaotic-map parameters to increase sensitivity. The full key required for decryption (including derived values) is exported alongside the ciphertext.
- OFF I/O: Faces are preserved; encryption operates on vertex coordinates. Encrypted meshes may contain extra vertices (padding) that are removed during decryption.

## Repository Layout
- [chaos_encrypt.py](chaos_encrypt.py): Core encryption/decryption (logistic, LDCML, tent maps; SHA-256 parameter derivation).
- [off_io.py](off_io.py): Minimal OFF reader/writer utilities.
- [metrics.py](metrics.py): Security metrics (entropy, correlation, NPCR/UACI, differential attack NPCR/UACI) and report printer.
- [demo.py](demo.py): CLI demo that runs the full pipeline on ModelNet10 or a single OFF file.
- [emd_eval.py](emd_eval.py): Compute Earth Mover's Distance (EMD) between outputs and originals.
- data/ModelNet10/ModelNet10: Expected dataset root (see Setup).
- output/: Encrypted/decrypted outputs and per-file keys (JSON).

## Requirements
- Python 3.8+
- NumPy

Install NumPy (if needed). Optionally install SciPy for exact EMD (faster and more precise than the greedy fallback):

```bash
pip install numpy
pip install scipy  # optional, for exact Hungarian matching in EMD
```

## Dataset Setup (ModelNet10)
This project expects the ModelNet10 dataset in OFF format at:

```
data/ModelNet10/ModelNet10/<class>/{train,test}/*.off
```

Place or symlink the dataset so each class directory (e.g., `bathtub`, `chair`, …) contains `train/` and `test/` folders with `.off` files. If you do not have the dataset, download the OFF version of ModelNet10 from the official source and arrange it in the structure above.

## Quick Start
Run the demo (process one example per class in the test split):

```powershell
python demo.py
```

Process a single OFF file:

```powershell
python demo.py --model "path\to\file.off"
```

Process all files in a split (slow):

```powershell
python demo.py --all --split test
```

Change key seeds (optional):

```powershell
python demo.py --u1 0.37891 --s1 0.45632 --beta 1.99964
```

Outputs are written to `output/<class>/`:
- `encrypted_<name>.off`: Encrypted mesh (may have extra vertices).
- `decrypted_<name>.off`: Decrypted mesh (should match original geometry).
- `key_<name>.json`: Full decryption key and metadata (includes plaintext-derived parameters and structural info like padding count and coordinate shift).

## How It Works (Brief)
- SHA-256 of the plaintext vertex matrix derives chaotic parameters (`n1..n3`, `alpha`, `alpha0`, `delta`) and the integer mask `bitxor_1` based on coordinate magnitudes.
- Logistic map sequence scrambles flattened coordinates; random rows (multiple of 3 values) are inserted to pad the ciphertext.
- LDCML outputs drive a permutation (reconfusion), an integer XOR-chain diffusion, and a fractional-part scramble.
- A tent-map sequence applies a final permutation.
- Decryption regenerates the same sequences from the stored key and reverses each step, then removes padding and undoes coordinate shifting.

See implementation details in [chaos_encrypt.py](chaos_encrypt.py).

## Security Metrics
The demo computes and prints a report with:
- Entropy of plaintext vs ciphertext.
- Adjacent-vertex correlation along x/y/z.
- Differential-attack NPCR/UACI (paper-style): compare ciphertexts from two plaintexts that differ by a tiny perturbation.
- Reconstruction error (max/mean) between plaintext and decrypted vertices.

Metrics are implemented in [metrics.py](metrics.py) and printed by [demo.py](demo.py).

### Earth Mover's Distance (EMD)
Evaluate geometric similarity between the original meshes and the results (encrypted/decrypted) using EMD on point clouds (vertex sets). By default, both point clouds are bbox-normalized and uniformly downsampled to at most 1024 points per set for performance. The default method uses the Sinkhorn (entropic-regularized) approximation.

Run across all available outputs:

```powershell
python emd_eval.py                  # encrypted+decrypted vs originals (Sinkhorn)
python emd_eval.py --subset decrypted
python emd_eval.py --subset encrypted
```

Common options:

- `--class <name>`: restrict to a single class (e.g., `chair`).
- `--max-points 2048`: increase matching size (slower, more accurate).
- `--no-normalize`: disable bbox normalization (use raw coordinates).
- `--method sinkhorn|hungarian|greedy`: choose algorithm (default: sinkhorn). `hungarian` requires SciPy and computes exact matching; `greedy` is a fast approximation.
- `--epsilon 0.05 --max-iter 200 --tol 1e-3`: Sinkhorn controls (smaller `epsilon` → closer to true EMD, potentially slower/less stable).

Results are printed per file and saved to `emd_results.csv`. If SciPy is installed, exact Hungarian matching is used; otherwise a greedy nearest-neighbor approximation is applied.

## Reproducibility & Keys
- The full key dictionary returned by encryption is saved as JSON per file. It contains both user-specified seeds (`u1`, `s1`, `beta`, `L`, `T`) and plaintext-derived parameters. Keep this file to enable decryption.
- Because parameters depend on the plaintext, encrypting numerically different inputs—even if visually similar—yields different keys and ciphertexts.

## Troubleshooting
- “No files found”: Verify your dataset path matches `data/ModelNet10/ModelNet10/<class>/{train,test}` and contains `.off` files.
- “Decryption error too large”: Ensure you are decrypting with the matching `key_*.json` produced for that ciphertext and file.
- Windows paths: Quote paths containing spaces when using `--model`.

## Reference
- Singh et al., “A novel 3-D image encryption algorithm based on SHA-256 and chaos theory,” Alexandria Engineering Journal, 2025.

If you use this code or adapt the approach, please cite the above work and acknowledge this repository.
