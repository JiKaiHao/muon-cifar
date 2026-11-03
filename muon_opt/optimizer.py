"""Single-device Muon with an AdamW auxiliary group for non-matrix parameters.

Matrix hidden weights (linear layers and convolutions, flattened to 2D) take a
momentum step and are then orthogonalized. Embeddings, heads, biases, and
normalization scales stay on AdamW, as in the original Muon setup.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from .newton_schulz import orthogonality_residual, scale_orthogonal_update, zeropower_via_newtonschulz5


def split_muon_params(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Hidden matrices go to Muon. The classifier, biases, and norm scales go to AdamW."""
    muon_params: list[torch.nn.Parameter] = []
    aux_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        head = name == "fc.weight" or name.endswith(".fc.weight") or name.endswith(".head.weight")
        if param.ndim >= 2 and not head:
            muon_params.append(param)
        else:
            aux_params.append(param)
    if not muon_params:
        raise ValueError("model has no matrix parameters for Muon")
    if not aux_params:
        raise ValueError("model has no auxiliary parameters for AdamW")
    return muon_params, aux_params


def _as_matrix(update: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """Flatten a convolution (out, in, kh, kw) into (out, in*kh*kw)."""
    if update.ndim == 2:
        return update, update.shape
    if update.ndim < 2:
        raise ValueError(f"Muon parameter must be at least 2D, got {tuple(update.shape)}")
    return update.reshape(update.shape[0], -1), update.shape


def _adam_update(
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    beta1, beta2 = betas
    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2)
    bias_c1 = 1 - beta1**step
    bias_c2 = 1 - beta2**step
    return (exp_avg / bias_c1) / (exp_avg_sq.sqrt() / bias_c2**0.5 + eps)


class MuonWithAuxAdam(Optimizer):
    """Muon for hidden matrices, AdamW for everything else.

    Muon parameter group keys: params, lr, momentum, weight_decay, ns_steps,
    adjust_lr, nesterov, use_muon=True.

    AdamW parameter group keys: params, lr, betas, eps, weight_decay, use_muon=False.
    """

    def __init__(self, param_groups: list[dict]):
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("each param group needs use_muon")
            if group["use_muon"]:
                group.setdefault("lr", 1e-3)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("ns_steps", 5)
                group.setdefault("adjust_lr", "match_rms_adamw")
                group.setdefault("nesterov", True)
            else:
                group.setdefault("lr", 1e-3)
                group.setdefault("betas", (0.9, 0.999))
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, defaults={})
        self._residual_sum = 0.0
        self._residual_count = 0
        self._rms_sum = 0.0
        self._rms_count = 0

    def pop_metrics(self) -> dict[str, float]:
        """Mean orthogonality residual and update RMS since the previous call."""
        residual = self._residual_sum / self._residual_count if self._residual_count else float("nan")
        rms = self._rms_sum / self._rms_count if self._rms_count else float("nan")
        self._residual_sum = 0.0
        self._residual_count = 0
        self._rms_sum = 0.0
        self._rms_count = 0
        return {"ortho_residual": residual, "update_rms": rms}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon(group)
            else:
                self._step_adam(group)
        return loss

    def _step_muon(self, group: dict) -> None:
        beta = group["momentum"]
        lr = group["lr"]
        wd = group["weight_decay"]
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            state = self.state[param]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(param)
            buf = state["momentum_buffer"]
            buf.lerp_(grad, 1 - beta)
            # Nesterov lookahead on the updated buffer: (1-beta) g + beta m.
            update = torch.lerp(grad, buf, beta) if group["nesterov"] else buf
            matrix, original_shape = _as_matrix(update)
            ortho = zeropower_via_newtonschulz5(matrix, steps=group["ns_steps"])
            residual = orthogonality_residual(ortho)
            scaled = scale_orthogonal_update(ortho, group["adjust_lr"]).reshape(original_shape)
            self._residual_sum += float(residual)
            self._residual_count += 1
            self._rms_sum += float(scaled.pow(2).mean().sqrt())
            self._rms_count += 1
            param.mul_(1 - lr * wd)
            param.add_(scaled.to(dtype=param.dtype), alpha=-lr)

    def _step_adam(self, group: dict) -> None:
        lr = group["lr"]
        wd = group["weight_decay"]
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            state = self.state[param]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] = 0
            state["step"] += 1
            update = _adam_update(
                grad, state["exp_avg"], state["exp_avg_sq"], state["step"], group["betas"], group["eps"]
            )
            param.mul_(1 - lr * wd)
            param.add_(update, alpha=-lr)
