"""
Security metrics for evaluating the 3D model encryption algorithm.

Computes:
  - Information Entropy  (Shannon entropy, theoretical max = 8)
  - Correlation Coefficient  along x, y, z directions
  - NPCR  (Number of Pixels Change Rate)
  - UACI  (Unified Average Changing Intensity)
"""

import numpy as np
from typing import Tuple, Optional

try:
    # Optional dependency: SciPy for exact Hungarian matching
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


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


# ---------------------------------------------------------------------------
# Earth Mover's Distance (EMD) / 1-Wasserstein on point clouds
# ---------------------------------------------------------------------------

def _normalize_points(points: np.ndarray, mode: str = 'bbox') -> np.ndarray:
    """
    Normalize a point cloud for scale/translation invariance.

    modes:
      - 'bbox': translate to bbox-center and scale so max side length == 1
      - 'none': no normalization
    """
    P = np.asarray(points, dtype=np.float64)
    if mode == 'none':
        return P
    mins = P.min(axis=0)
    maxs = P.max(axis=0)
    center = (mins + maxs) / 2.0
    scale = float(np.max(maxs - mins))
    if scale <= 0:
        scale = 1.0
    return (P - center) / scale


def _downsample_equal(P: np.ndarray, Q: np.ndarray, max_points: int,
                      seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample both sets to the same size n = min(len(P), len(Q), max_points)."""
    n = int(min(len(P), len(Q), max_points))
    if n <= 0:
        return P[:0], Q[:0]
    rng = np.random.default_rng(seed)
    idxP = rng.choice(len(P), size=n, replace=False) if len(P) > n else np.arange(len(P))
    idxQ = rng.choice(len(Q), size=n, replace=False) if len(Q) > n else np.arange(len(Q))
    # If one side had fewer points than n (shouldn't happen due to min), adjust
    n = min(len(idxP), len(idxQ))
    return P[idxP[:n]], Q[idxQ[:n]]


def emd_point_cloud(P: np.ndarray,
                    Q: np.ndarray,
                    normalize: str = 'bbox',
                    max_points: int = 1024,
                    seed: int = 0,
                    exact_if_scipy: bool = True) -> float:
    """
    Compute Earth Mover's Distance (1-Wasserstein with L2 ground metric)
    between two 3D point clouds with uniform weights.

    - If SciPy is available and `exact_if_scipy` is True, uses the Hungarian
      algorithm for an exact minimum-cost matching (O(n^3)).
    - Otherwise uses a greedy nearest-neighbor matching (approximation).

    Parameters
    ----------
    P, Q        : np.ndarray of shape (N, 3) and (M, 3)
    normalize   : 'bbox' or 'none' – preprocessing to make distances comparable
    max_points  : cap number of points (for performance). Downsamples uniformly.
    seed        : RNG seed for reproducible downsampling
    exact_if_scipy : use SciPy Hungarian if available

    Returns
    -------
    emd : float – average transport cost (lower is more similar)
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.ndim != 2 or Q.ndim != 2 or P.shape[1] != 3 or Q.shape[1] != 3:
        raise ValueError("P and Q must be (N,3) and (M,3) arrays")

    # Normalize
    Pn = _normalize_points(P, mode=normalize)
    Qn = _normalize_points(Q, mode=normalize)

    # Downsample equally
    Pn, Qn = _downsample_equal(Pn, Qn, max_points=max_points, seed=seed)
    n = len(Pn)
    if n == 0:
        return 0.0

    # Cost matrix (Euclidean distances)
    # For performance, compute squared distances in blocks if needed; here n<=max_points
    # so a dense compute is fine.
    diff = Pn[:, None, :] - Qn[None, :, :]
    C = np.sqrt(np.sum(diff * diff, axis=2))  # shape (n, n)

    if exact_if_scipy and _HAVE_SCIPY:
        row_ind, col_ind = linear_sum_assignment(C)
        emd = float(C[row_ind, col_ind].mean())
        return emd

    # Greedy fallback: iteratively match nearest unmatched pairs
    emd_sum = 0.0
    used = np.zeros(n, dtype=bool)
    for i in range(n):
        # nearest Q for P[i] among unused
        dists = C[i].copy()
        dists[used] = np.inf
        j = int(np.argmin(dists))
        used[j] = True
        emd_sum += float(dists[j])
    return emd_sum / n


def sinkhorn_emd_point_cloud(P: np.ndarray,
                             Q: np.ndarray,
                             normalize: str = 'bbox',
                             max_points: int = 1024,
                             seed: int = 0,
                             epsilon: float = 0.05,
                             max_iter: int = 200,
                             tol: float = 1e-3) -> float:
    """
    Compute entropically-regularized OT (Sinkhorn distance) between two
    point clouds with uniform weights using the Sinkhorn-Knopp algorithm.

    Returns the transport cost sum(P * C) where C is the pairwise Euclidean
    distance matrix. With bbox normalization, distances are scale-invariant.

    Parameters
    ----------
    P, Q       : (N,3), (M,3) point clouds
    normalize  : 'bbox' or 'none'
    max_points : cap on points per cloud (uniform downsampling)
    seed       : RNG seed for downsampling
    epsilon    : entropic regularization strength (>0). Smaller -> closer to EMD, but harder numerically.
    max_iter   : maximum Sinkhorn iterations
    tol        : stopping tolerance on marginal errors (L1 per row/col)
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.ndim != 2 or Q.ndim != 2 or P.shape[1] != 3 or Q.shape[1] != 3:
        raise ValueError("P and Q must be (N,3) and (M,3) arrays")

    # Normalize and downsample equally
    Pn = _normalize_points(P, mode=normalize)
    Qn = _normalize_points(Q, mode=normalize)
    Pn, Qn = _downsample_equal(Pn, Qn, max_points=max_points, seed=seed)
    n = len(Pn)
    if n == 0:
        return 0.0

    # Cost matrix (Euclidean distances)
    diff = Pn[:, None, :] - Qn[None, :, :]
    C = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)

    # Uniform marginals (sum to 1)
    a = np.full(n, 1.0 / n)
    b = np.full(n, 1.0 / n)

    # Gibbs kernel K = exp(-C/epsilon)
    # Clip exponent for numerical stability
    E = -C / max(epsilon, 1e-6)
    E = np.clip(E, -60.0, 60.0)
    K = np.exp(E) + 1e-12

    # Sinkhorn iterations: u <- a / (K v), v <- b / (K^T u)
    u = np.ones(n)
    v = np.ones(n)
    for _ in range(max_iter):
        Kv = K @ v
        # Avoid divide by zero
        Kv = np.maximum(Kv, 1e-12)
        u_new = a / Kv

        KTu = K.T @ u_new
        KTu = np.maximum(KTu, 1e-12)
        v_new = b / KTu

        # Check marginal errors occasionally
        if _ % 10 == 0:
            # Current transport plan rowsums and colsums
            # rowsums = u_new * (K @ v_new)
            rowsums = u_new * (K @ v_new)
            colsums = v_new * (K.T @ u_new)
            err = float(np.mean(np.abs(rowsums - a)) + np.mean(np.abs(colsums - b)))
            if err < tol:
                u, v = u_new, v_new
                break
        u, v = u_new, v_new

    # Transport plan P = diag(u) K diag(v)
    # Compute cost = sum(P * C)
    # To avoid forming full P explicitly (n<=max_points so it's fine), but compute efficiently
    P_plan = (u[:, None] * K) * v[None, :]
    cost = float(np.sum(P_plan * C))
    return cost
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
