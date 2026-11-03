"""Checks that Newton–Schulz approaches the polar factor and that one Muon step runs."""

from __future__ import annotations

import torch

from muon_opt import MuonWithAuxAdam, orthogonality_residual, polar_factor, zeropower_via_newtonschulz5
from muon_opt.newton_schulz import scale_orthogonal_update


def check_newton_schulz(device: torch.device) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    matrix = torch.randn(64, 32, generator=generator, device="cpu").to(device)
    target = polar_factor(matrix)
    distances = []
    for steps in (1, 3, 5):
        approx = zeropower_via_newtonschulz5(matrix, steps=steps).float()
        distances.append(float((approx - target).norm() / target.norm()))
        residual = float(orthogonality_residual(approx))
        print(f"ns_steps={steps}  polar_rel={distances[-1]:.4f}  ortho_residual={residual:.4f}")
    if not distances[2] < distances[0]:
        raise AssertionError(f"5 Newton–Schulz steps should beat 1 step, got {distances}")
    scaled = scale_orthogonal_update(zeropower_via_newtonschulz5(matrix, steps=5), "match_rms_adamw")
    rms = float(scaled.pow(2).mean().sqrt())
    print(f"rms_after_moonlight_scale={rms:.4f}")
    if not 0.05 < rms < 0.5:
        raise AssertionError(f"Moonlight scale should put RMS near 0.2, got {rms}")


def check_step(device: torch.device) -> None:
    torch.manual_seed(0)
    layer = torch.nn.Linear(16, 8, bias=True).to(device)
    head = torch.nn.Linear(8, 4, bias=True).to(device)
    opt = MuonWithAuxAdam(
        [
            {
                "params": [layer.weight],
                "lr": 1e-2,
                "momentum": 0.95,
                "weight_decay": 0.0,
                "ns_steps": 5,
                "adjust_lr": "match_rms_adamw",
                "nesterov": True,
                "use_muon": True,
            },
            {
                "params": [layer.bias, head.weight, head.bias],
                "lr": 1e-2,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "use_muon": False,
            },
        ]
    )
    before = layer.weight.detach().clone()
    loss_value = None
    for _ in range(20):
        x = torch.randn(32, 16, device=device)
        y = torch.randint(0, 4, (32,), device=device)
        opt.zero_grad(set_to_none=True)
        hidden = torch.relu(layer(x))
        loss = torch.nn.functional.cross_entropy(head(hidden), y)
        loss.backward()
        opt.step()
        loss_value = float(loss)
    delta = float((layer.weight - before).norm())
    metrics = opt.pop_metrics()
    print(f"step_delta={delta:.4f}  last_loss={loss_value:.4f}  metrics={metrics}")
    if delta == 0.0:
        raise AssertionError("Muon step did not move the matrix weight")
    if metrics["ortho_residual"] != metrics["ortho_residual"]:
        raise AssertionError("orthogonality residual was not recorded")


def check_conv(device: torch.device) -> None:
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False).to(device)
    norm = torch.nn.BatchNorm2d(16).to(device)
    opt = MuonWithAuxAdam(
        [
            {
                "params": [conv.weight],
                "lr": 0.02,
                "momentum": 0.95,
                "weight_decay": 1e-4,
                "ns_steps": 5,
                "adjust_lr": "original",
                "nesterov": True,
                "use_muon": True,
            },
            {
                "params": [norm.weight, norm.bias],
                "lr": 1e-3,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 1e-4,
                "use_muon": False,
            },
        ]
    )
    before = conv.weight.detach().clone()
    x = torch.randn(8, 3, 32, 32, device=device)
    y = torch.randn(8, 16, 32, 32, device=device)
    opt.zero_grad(set_to_none=True)
    pred = norm(conv(x))
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    opt.step()
    delta = float((conv.weight - before).norm())
    print(f"conv_delta={delta:.4f}  loss={float(loss):.4f}")
    if delta == 0.0:
        raise AssertionError("Muon did not update the convolution")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    check_newton_schulz(device)
    check_step(device)
    check_conv(device)
    print("ok")


if __name__ == "__main__":
    main()
