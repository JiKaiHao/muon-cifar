"""Newton–Schulz orthogonalization used by Muon.

The quintic coefficients and the wide-matrix iteration follow Keller Jordan's
reference implementation (https://github.com/KellerJordan/Muon). The iteration
approximates the polar factor UV^T of a matrix, which is the steepest-descent
direction under the spectral norm. The published coefficients intentionally
keep a steep slope at zero, so singular values land near Uniform(0.5, 1.5)
rather than exactly 1.
"""

from __future__ import annotations

import torch

# Selected to maximize the slope at zero of the quintic Newton–Schulz map.
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the polar factor of a matrix (or a batch of matrices).

    ``steps=0`` returns G normalized to spectral norm at most 1, with no iteration.
    The Gram matrix is formed on the shorter side, matching the reference code.
    """
    if G.ndim < 2:
        raise ValueError(f"Newton–Schulz expects a matrix, got shape {tuple(G.shape)}")
    a, b, c = NS_COEFFS
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        gram = X @ X.mT
        gram_poly = b * gram + c * gram @ gram
        X = a * X + gram_poly @ X
    if transposed:
        X = X.mT
    return X


def scale_orthogonal_update(X: torch.Tensor, mode: str) -> torch.Tensor:
    """Rescale an orthogonalized update.

    ``original`` is Keller Jordan's aspect-ratio correction, sqrt(max(1, A/B)),
    applied to the matrix Newton–Schulz actually saw (convolutions are flattened
    first). ``match_rms_adamw`` is the Moonlight correction, 0.2 * sqrt(max(A, B)),
    which puts the update RMS near the empirical RMS of AdamW so the same learning
    rate can be reused. ``none`` leaves the orthogonalized matrix unscaled.
    """
    rows, cols = X.size(-2), X.size(-1)
    if mode == "original":
        coef = max(1.0, rows / cols) ** 0.5
    elif mode == "match_rms_adamw":
        coef = 0.2 * max(rows, cols) ** 0.5
    elif mode == "none":
        coef = 1.0
    else:
        raise ValueError(f"unknown adjust_lr mode: {mode}")
    return X.float() * coef


def orthogonality_residual(X: torch.Tensor) -> torch.Tensor:
    """Relative Frobenius residual of U^T U = I (or U U^T = I) on the short side."""
    matrix = X.float()
    if matrix.size(-2) >= matrix.size(-1):
        gram = matrix.mT @ matrix
    else:
        gram = matrix @ matrix.mT
    eye = torch.eye(gram.size(-1), device=gram.device, dtype=gram.dtype)
    return (gram - eye).norm() / eye.norm()


def polar_factor(G: torch.Tensor) -> torch.Tensor:
    """Exact polar factor UV^T from the thin SVD, for tests and diagnostics."""
    matrix = G.float()
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    return u @ vh
