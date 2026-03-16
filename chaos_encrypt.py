"""
Chaos-based 3D model encryption algorithm.

Implements: "A novel 3-D image encryption algorithm based on SHA-256 and chaos theory"
(Singh et al., Alexandria Engineering Journal, 2025)

Adapted to work with ModelNet10 OFF format instead of STL.

Algorithm stages:
  Stage 1 – Logistic map: scramble vertex coordinates + insert random points
  Stage 2 – LDCML:       reconfuse and diffuse (integer XOR-chain + fractional scramble)
  Stage 3 – Tent map:    final confusion permutation
"""

import numpy as np
import hashlib


# ---------------------------------------------------------------------------
# Chaotic maps
# ---------------------------------------------------------------------------

def _logistic_iterate(u0: float, alpha: float, n: int) -> np.ndarray:
    """Iterate the logistic map  u_{i+1} = alpha * u_i * (1 - u_i)  n times."""
    seq = np.empty(n, dtype=np.float64)
    seq[0] = u0
    for i in range(1, n):
        seq[i] = alpha * seq[i - 1] * (1.0 - seq[i - 1])
    return seq


def _ldcml_iterate(n1: float, n2: float, n3: float,
                   alpha: float, alpha0: float, delta: float,
                   total_iters: int) -> np.ndarray:
    """
    Logistic-Dynamic Coupled Logistic Map Lattice (LDCML) with L=3 cells.

    u_{n+1}(k) = (1 - L(δ)) f(u_n(k))
                + (L(δ)/2)  f(u_n(k-1))
                +            f(u_n(k+1))
    where L(δ) = α0 * δ * (1 - δ),  f(u) = alpha * u * (1 - u).
    Boundary: periodic (k-1=L-1 when k=0, k+1=0 when k=L-1).

    Returns array of shape (total_iters, 3).
    """
    u = np.array([n1, n2, n3], dtype=np.float64)
    L = 3
    results = np.empty((total_iters, 3), dtype=np.float64)
    Ld = alpha0 * delta * (1.0 - delta)

    for step in range(total_iters):
        results[step] = u
        # Clamp u to valid chaotic range before each logistic evaluation
        u = np.clip(u, 1e-10, 1.0 - 1e-10)
        f = alpha * u * (1.0 - u)          # logistic applied element-wise
        u_new = np.empty(3, dtype=np.float64)
        for k in range(L):
            k_prev = (k - 1) % L
            k_next = (k + 1) % L
            # Standard LDCML: coefficients sum to 1 (convex combination)
            u_new[k] = ((1.0 - Ld) * f[k]
                        + (Ld / 2.0) * f[k_prev]
                        + (Ld / 2.0) * f[k_next])
        u = u_new

    return results


def _tent_iterate(s1: float, beta: float, n: int) -> np.ndarray:
    """Iterate the tent map  n times from initial value s1 ∈ (0,1)."""
    seq = np.empty(n, dtype=np.float64)
    seq[0] = s1
    for i in range(1, n):
        if seq[i - 1] < 0.5:
            seq[i] = beta * seq[i - 1]
        else:
            seq[i] = beta * (1.0 - seq[i - 1])
    return seq


# ---------------------------------------------------------------------------
# Helper: bitxor_1
# ---------------------------------------------------------------------------

def _compute_bitxor1(X: np.ndarray) -> int:
    """
    bitxor_1 = 2^n - 1, where n is the bit-length of floor(max(|X|)).
    This is the all-ones mask with the same number of bits as the
    integer part of the largest coordinate value.
    """
    max_int = int(np.floor(np.max(np.abs(X))))
    if max_int == 0:
        return 1
    n_bits = int(np.floor(np.log2(max_int))) + 1
    return (1 << n_bits) - 1


# ---------------------------------------------------------------------------
# Key generation (from SHA-256 of plaintext)
# ---------------------------------------------------------------------------

def _sha256_derive_params(X: np.ndarray):
    """
    Apply SHA-256 to the coordinate matrix X and derive:
      n1, n2, n3  – initial conditions for LDCML (in [0,1])
      alpha        – logistic map parameter (≈3.99)
      alpha0       – LDCML logistic coupling parameter (≈3.99)
      delta        – LDCML coupling coefficient (≈0.01…0.10)
    """
    X_bytes = X.astype(np.float64).tobytes()
    h_hex = hashlib.sha256(X_bytes).hexdigest()
    h_int = int(h_hex, 16)
    h_bits = format(h_int, '0256b')        # 256-bit binary string

    # Take first 240 bits, split into 6 × 40-bit subkeys
    h_240 = h_bits[:240]
    subkeys = [int(h_240[i * 40:(i + 1) * 40], 2) / (2 ** 40)
               for i in range(6)]

    n1, n2, n3 = subkeys[0], subkeys[1], subkeys[2]
    n4, n5, n6 = subkeys[3], subkeys[4], subkeys[5]

    alpha  = 3.99 + 0.01 * n4
    alpha0 = 3.99 + 0.01 * n5
    delta  = 0.01 + 0.09 * n6

    # Clamp ICs away from logistic-map fixed points {0, 1}
    eps = 1e-6
    n1 = max(eps, min(1.0 - eps, n1))
    n2 = max(eps, min(1.0 - eps, n2))
    n3 = max(eps, min(1.0 - eps, n3))

    return n1, n2, n3, alpha, alpha0, delta


# ---------------------------------------------------------------------------
# Inverse permutation helper
# ---------------------------------------------------------------------------

def _inv_perm(idx: np.ndarray) -> np.ndarray:
    """Return the inverse of a permutation given as an index array."""
    inv = np.empty_like(idx)
    inv[idx] = np.arange(len(idx))
    return inv


# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------

def encrypt(vertices: np.ndarray, key: dict) -> tuple:
    """
    Encrypt 3-D model vertex coordinates.

    Parameters
    ----------
    vertices : np.ndarray, shape (s, 3)
        Plaintext vertex coordinate matrix.
    key : dict with mandatory fields:
        u1   – initial scalar for logistic map seed (0 < u1 < 1)
        s1   – initial value for tent map (0 < s1 < 1)
        beta – tent map control parameter (1 < beta < 2)
        L    – logistic map transient length (default 100)
        T    – LDCML transient length (default 100)

    Returns
    -------
    enc_vertices : np.ndarray, shape (s', 3)   – encrypted (+ padded) vertices
    enc_key      : dict  – full key needed for decryption (store securely)
    """
    u1   = float(key['u1'])
    s1   = float(key['s1'])
    beta = float(key['beta'])
    L    = int(key.get('L', 100))
    T    = int(key.get('T', 100))

    s = len(vertices)   # original number of vertices
    X_orig = vertices.astype(np.float64).copy()

    # ── Shift coordinates to positive (algorithm requires positive floats) ──
    coord_min = float(np.min(X_orig))
    shift = max(0.0, -coord_min + 1e-6)
    X = X_orig + shift          # all values now > 0

    # ── Step 1: bitxor_1 ────────────────────────────────────────────────────
    bitxor_1 = _compute_bitxor1(X)

    # ── Step 2: Derive parameters from SHA-256 of plaintext ─────────────────
    n1, n2, n3, alpha, alpha0, delta = _sha256_derive_params(X)

    # ── Step 3: Logistic map sequence B ─────────────────────────────────────
    u0 = abs(u1 - (float(np.sum(np.floor(X))) % 100) / 1000.0)
    log_seq = _logistic_iterate(u0, alpha, s * 3 + L)
    B = log_seq[L:]             # discard transient; |B| = s*3

    # ── Step 4: Confusion – scramble X with argsort(B) ──────────────────────
    X_flat = X.flatten()        # shape: (s*3,)
    idx1   = np.argsort(B)
    X1_flat = X_flat[idx1]      # scrambled

    # ── Step 5: Insert random rows ───────────────────────────────────────────
    # random is always a multiple of 9 so that random/3 is a multiple of 3
    random = int(np.floor((s * 3 * u1) / 100.0 + 1)) * 9
    # Clamp to at most half of B length
    max_rand = (len(B) // 2 - 1)
    max_rand = (max_rand // 9) * 9
    random = min(random, max_rand)
    if random < 9:
        random = 9

    B_for_Y = B[:random]               # first `random` values
    Y0      = B[random: 2 * random]    # next  `random` values
    Y1 = np.floor(bitxor_1 * B_for_Y) + np.round(Y0, 4)
    Y2 = Y1.reshape(random // 3, 3)    # shape: (random/3, 3)

    X1 = X1_flat.reshape(s, 3)
    X2 = np.vstack([X1, Y2])           # shape: (s + random//3, 3)

    total_size = s * 3 + random        # == X2.size

    # ── Step 6: LDCML sequences D1, D2, D3 ──────────────────────────────────
    ldcml_out = _ldcml_iterate(n1, n2, n3, alpha, alpha0, delta,
                               total_size + T)
    D1 = ldcml_out[T:, 0]   # shape: (total_size,)
    D2 = ldcml_out[T:, 1]
    D3 = ldcml_out[T:, 2]

    # ── Step 7: Reconfusion with D1 ─────────────────────────────────────────
    X2_flat = X2.flatten()
    idx2    = np.argsort(D1)
    X3_flat = X2_flat[idx2]

    # ── Step 8: Diffusion ────────────────────────────────────────────────────
    X4_flat = np.mod(X3_flat, 1.0)                     # fractional part
    X5_flat = np.floor(X3_flat).astype(np.int64)        # integer part

    # Key stream from D2
    P = np.mod(np.floor(D2 * 1e14).astype(np.int64), bitxor_1)

    # XOR-chain diffusion on integer part
    Q = np.empty_like(X5_flat)
    Q[0] = int(X5_flat[0]) ^ int(P[0])
    for i in range(1, len(X5_flat)):
        Q[i] = int(X5_flat[i]) ^ int(Q[i - 1]) ^ int(P[i])

    # Scramble fractional part with D3
    idx3    = np.argsort(D3)
    S1_flat = X4_flat[idx3]

    # Combine: X6 = Q (encrypted integer) + S1 (scrambled fractional)
    X6_flat = Q.astype(np.float64) + S1_flat

    # ── Step 9: Final confusion with tent map ────────────────────────────────
    tent_seq   = _tent_iterate(s1, beta, total_size)
    idx_tent   = np.argsort(tent_seq)
    X7_flat    = X6_flat[idx_tent]

    enc_vertices = X7_flat.reshape(-1, 3)

    # Full key (needed for decryption – include plaintext-derived params)
    enc_key = {
        # User-supplied
        'u1': u1, 's1': s1, 'beta': beta, 'L': L, 'T': T,
        # Derived from plaintext (must travel with ciphertext)
        'n1': n1, 'n2': n2, 'n3': n3,
        'alpha': alpha, 'alpha0': alpha0, 'delta': delta,
        'u0': u0,
        'bitxor_1': bitxor_1,
        # Structural metadata
        'coord_shift': shift,
        'original_s': s,
        'random': random,
    }

    return enc_vertices, enc_key


# ---------------------------------------------------------------------------
# Decryption
# ---------------------------------------------------------------------------

def decrypt(enc_vertices: np.ndarray, enc_key: dict) -> np.ndarray:
    """
    Decrypt 3-D model vertex coordinates.

    Parameters
    ----------
    enc_vertices : np.ndarray  – encrypted vertex matrix (from encrypt())
    enc_key      : dict        – key returned by encrypt()

    Returns
    -------
    vertices : np.ndarray, shape (s, 3) – recovered plaintext vertices
    """
    u1       = float(enc_key['u1'])
    s1       = float(enc_key['s1'])
    beta     = float(enc_key['beta'])
    L        = int(enc_key['L'])
    T        = int(enc_key['T'])
    n1       = float(enc_key['n1'])
    n2       = float(enc_key['n2'])
    n3       = float(enc_key['n3'])
    alpha    = float(enc_key['alpha'])
    alpha0   = float(enc_key['alpha0'])
    delta    = float(enc_key['delta'])
    u0       = float(enc_key['u0'])
    bitxor_1 = int(enc_key['bitxor_1'])
    shift    = float(enc_key['coord_shift'])
    s        = int(enc_key['original_s'])
    random   = int(enc_key['random'])

    total_size = s * 3 + random

    X7_flat = enc_vertices.flatten()

    # ── Inverse Step 9: inverse tent map permutation ─────────────────────────
    tent_seq = _tent_iterate(s1, beta, total_size)
    idx_tent = np.argsort(tent_seq)
    inv_tent = _inv_perm(idx_tent)
    X6_flat  = X7_flat[inv_tent]

    # ── Inverse Step 8: Diffusion ────────────────────────────────────────────
    # Separate integer (Q) and fractional (S1) parts
    Q      = np.floor(X6_flat).astype(np.int64)
    S1_flat = np.mod(X6_flat, 1.0)

    # Regenerate LDCML sequences
    ldcml_out = _ldcml_iterate(n1, n2, n3, alpha, alpha0, delta,
                               total_size + T)
    D1 = ldcml_out[T:, 0]
    D2 = ldcml_out[T:, 1]
    D3 = ldcml_out[T:, 2]

    # Recover fractional part X4 (inverse of D3-scramble)
    idx3    = np.argsort(D3)
    inv_idx3 = _inv_perm(idx3)
    X4_flat  = S1_flat[inv_idx3]

    # Recover integer part X5 via inverse XOR-chain
    P = np.mod(np.floor(D2 * 1e14).astype(np.int64), bitxor_1)
    X5_flat = np.empty_like(Q)
    X5_flat[0] = int(Q[0]) ^ int(P[0])
    for i in range(1, len(Q)):
        X5_flat[i] = int(Q[i]) ^ int(Q[i - 1]) ^ int(P[i])

    # Reassemble X3
    X3_flat = X5_flat.astype(np.float64) + X4_flat

    # ── Inverse Step 7: inverse D1-scramble ──────────────────────────────────
    idx2     = np.argsort(D1)
    inv_idx2 = _inv_perm(idx2)
    X2_flat  = X3_flat[inv_idx2]

    # ── Remove random rows (keep only first s rows = s*3 values) ────────────
    X1_flat = X2_flat[:s * 3]

    # ── Inverse Step 4: inverse B-scramble ───────────────────────────────────
    log_seq = _logistic_iterate(u0, alpha, s * 3 + L)
    B       = log_seq[L:]
    idx1    = np.argsort(B)
    inv_idx1 = _inv_perm(idx1)
    X_flat  = X1_flat[inv_idx1]

    # Reshape and undo coordinate shift
    X_dec = X_flat.reshape(s, 3) - shift
    return X_dec
