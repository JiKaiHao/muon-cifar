"""Single-device Muon: momentum orthogonalized by Newton–Schulz."""

from .newton_schulz import orthogonality_residual, polar_factor, zeropower_via_newtonschulz5
from .optimizer import MuonWithAuxAdam, split_muon_params

__all__ = [
    "MuonWithAuxAdam",
    "orthogonality_residual",
    "polar_factor",
    "split_muon_params",
    "zeropower_via_newtonschulz5",
]
