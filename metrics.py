"""
Security metrics for evaluating the 3D model encryption algorithm.

Computes:
  - Information Entropy  (Shannon entropy, theoretical max = 8)
  - Correlation Coefficient  along x, y, z directions
  - NPCR  (Number of Pixels Change Rate)
  - UACI  (Unified Average Changing Intensity)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Information Entropy
# ---------------------------------------------------------------------------

def information_entropy(coords: np.ndarray, n_bins: int = 256) -> float:
    """
    Shannon entropy of the flattened coordinate array.

    The paper partitions the floating-point values into n_bins equal-width
    bins over [min, max) and computes the empirical entropy.  A perfect
    random distribution gives log2(256) = 8.0.

    Parameters
    ----------
    coords  : np.ndarray – vertex coordinate matrix (any shape)
    n_bins  : int        – number of histogram bins (default 256)
    """
    flat = coords.flatten()
    v_min, v_max = flat.min(), flat.max()
    if v_max == v_min:
        return 0.0

    # Bin the data
    counts, _ = np.histogram(flat, bins=n_bins, range=(v_min, v_max))
    probs = counts / counts.sum()
    probs = probs[probs > 0]                     # avoid log(0)
    return float(-np.sum(probs * np.log2(probs)))


# ---------------------------------------------------------------------------
# Correlation Coefficient
# ---------------------------------------------------------------------------

def correlation_adjacent(coords: np.ndarray) -> dict:
    """
    Correlation coefficient between adjacent vertex values along each
    coordinate axis (x, y, z).

    For each axis, paired samples are (coords[i, axis], coords[i+1, axis])
    for i in 0..n-2.

    Returns a dict with keys 'x', 'y', 'z'.
    """
    result = {}
    axes = ['x', 'y', 'z']
    for j, ax in enumerate(axes):
        u = coords[:-1, j]
        v = coords[1:,  j]
        mean_u = u.mean()
        mean_v = v.mean()
        cov    = np.mean((u - mean_u) * (v - mean_v))
        std_u  = np.sqrt(np.mean((u - mean_u) ** 2))
        std_v  = np.sqrt(np.mean((v - mean_v) ** 2))
        if std_u * std_v == 0:
            result[ax] = 0.0
        else:
            result[ax] = float(cov / (std_u * std_v))
    return result


# ---------------------------------------------------------------------------
# NPCR and UACI
# ---------------------------------------------------------------------------

def npcr_uaci(plain: np.ndarray, cipher: np.ndarray,
              value_range: float = None) -> tuple:
    """
    NPCR – Number of Pixels/Points Change Rate (should be close to 100%).
    UACI – Unified Average Changing Intensity (should be close to 33.33%).

    Both are computed on the flattened coordinate arrays.

    Parameters
    ----------
    plain   : np.ndarray – plaintext coordinate matrix
    cipher  : np.ndarray – ciphertext coordinate matrix (same shape)
    value_range : float  – theoretical value range (default: max - min of plain)

    Returns
    -------
    npcr_val : float  in [0, 100]  (%)
    uaci_val : float  in [0, 100]  (%)
    """
    p = plain.flatten().astype(np.float64)
    c = cipher.flatten().astype(np.float64)
    n = len(p)

    if value_range is None:
        value_range = float(p.max() - p.min())
        if value_range == 0:
            value_range = 1.0

    # NPCR: fraction of positions that changed
    changed = np.sum(p != c)
    npcr_val = 100.0 * changed / n

    # UACI: average normalised change magnitude
    uaci_val = 100.0 * np.sum(np.abs(p - c)) / (n * value_range)

    return float(npcr_val), float(uaci_val)


def npcr_uaci_paper(cipher_a: np.ndarray,
                    cipher_b: np.ndarray,
                    intensity_denominator: float = 255.0) -> tuple:
    """
    Differential-attack NPCR/UACI as described in Section 5.7 of the paper:
      1) Encrypt two plaintexts that differ slightly.
      2) Compare resulting ciphertexts.

    NPCR is computed as position-wise inequality ratio over ciphertext arrays.
    UACI is computed as the average absolute difference normalized by
    intensity_denominator.

    In the original paper data, values are in [0,256), so 255 is used.
    For ModelNet adaptation, passing bitxor_1 is often more representative of
    the effective ciphertext intensity range.
    """
    a = np.asarray(cipher_a, dtype=np.float64).reshape(-1)
    b = np.asarray(cipher_b, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        m = min(a.size, b.size)
        a = a[:m]
        b = b[:m]

    n = a.size
    npcr_val = 100.0 * np.sum(a != b) / n
    denom = float(intensity_denominator) if intensity_denominator else 255.0
    if denom <= 0:
        denom = 255.0
    uaci_val = 100.0 * np.sum(np.abs(a - b)) / (n * denom)
    return float(npcr_val), float(uaci_val)


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def security_report(plain_coords: np.ndarray,
                    enc_coords: np.ndarray,
                    dec_coords: np.ndarray) -> dict:
    """
    Compute and return a full security-analysis report.

    Parameters
    ----------
    plain_coords : plaintext vertex matrix
    enc_coords   : encrypted vertex matrix  (may have more rows due to padding)
    dec_coords   : decrypted vertex matrix  (should match plain_coords)

    Returns
    -------
    report : dict with all metrics
    """
    report = {}

    # Entropy
    report['entropy_plain']  = information_entropy(plain_coords)
    report['entropy_cipher'] = information_entropy(enc_coords)

    # Correlation
    report['corr_plain']  = correlation_adjacent(plain_coords)
    report['corr_cipher'] = correlation_adjacent(enc_coords[:len(plain_coords)])

    # Legacy NPCR/UACI: plain vs cipher (kept for comparison only)
    enc_trimmed = enc_coords[:len(plain_coords)]
    report['npcr_plain_cipher'], report['uaci_plain_cipher'] = npcr_uaci(plain_coords, enc_trimmed)

    # Reconstruction error
    if dec_coords is not None:
        max_err  = float(np.max(np.abs(plain_coords - dec_coords)))
        mean_err = float(np.mean(np.abs(plain_coords - dec_coords)))
        report['max_reconstruction_error']  = max_err
        report['mean_reconstruction_error'] = mean_err

    return report


def print_report(report: dict):
    """Pretty-print the security report."""
    print("=" * 55)
    print("          SECURITY ANALYSIS REPORT")
    print("=" * 55)

    print(f"\n[Entropy]")
    print(f"  Plaintext  entropy : {report['entropy_plain']:.4f}  (ideal ≈ 7-8)")
    print(f"  Ciphertext entropy : {report['entropy_cipher']:.4f}  (ideal ≈ 8.0)")

    print(f"\n[Correlation – adjacent vertices]")
    cp = report['corr_plain']
    cc = report['corr_cipher']
    for ax in ('x', 'y', 'z'):
        print(f"  {ax}-axis  plain={cp[ax]:+.4f}   cipher={cc[ax]:+.4f}  (ideal cipher ≈ 0)")

    print(f"\n[Differential Attack]")
    if 'npcr_diff' in report and 'uaci_diff' in report:
        print(f"  NPCR : {report['npcr_diff']:.4f} %  (paper-style: C1 vs C2)")
        print(f"  UACI : {report['uaci_diff']:.4f} %  (paper-style: C1 vs C2)")
    else:
        print(f"  NPCR : {report['npcr_plain_cipher']:.4f} %  (plain vs cipher)")
        print(f"  UACI : {report['uaci_plain_cipher']:.4f} %  (plain vs cipher)")

    if 'max_reconstruction_error' in report:
        print(f"\n[Decryption Fidelity]")
        print(f"  Max  error : {report['max_reconstruction_error']:.2e}")
        print(f"  Mean error : {report['mean_reconstruction_error']:.2e}")

    print("=" * 55)
