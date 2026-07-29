from __future__ import annotations

import sys
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.pfe_unet import PFEUNet as MultiscalePFEUNet
from models.pfe_unet_reference import PFEUNet as ReferencePFEUNet


CORE_VARIANTS = {
    "M0": (),
    "M1": (1,),
    "M2": (3,),
    "M3": (5,),
    "M4": (1, 2, 3),
    "M5": (3, 4, 5),
    "M6": (1, 2, 3, 4, 5),
}


def assert_all_five_matches_reference() -> None:
    torch.manual_seed(17)
    reference = ReferencePFEUNet(
        in_channels=4,
        out_channels=1,
        base_channels=8,
        enhancement="all",
        fusion="multiply",
    ).eval()
    candidate = MultiscalePFEUNet(
        in_channels=4,
        out_channels=1,
        base_channels=8,
        enhancement="all",
        fusion="multiply",
        active_gate_levels=CORE_VARIANTS["M6"],
    ).eval()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    sample = torch.randn(2, 4, 64, 64)
    with torch.no_grad():
        expected = reference(sample)
        actual = candidate(sample)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def assert_no_gate_is_polarization_invariant() -> None:
    torch.manual_seed(23)
    model = MultiscalePFEUNet(
        in_channels=4,
        out_channels=1,
        base_channels=8,
        enhancement="all",
        fusion="multiply",
        active_gate_levels=CORE_VARIANTS["M0"],
    ).eval()
    rgb = torch.randn(2, 3, 64, 64)
    polar_a = torch.zeros(2, 1, 64, 64)
    polar_b = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        output_a = model(torch.cat([rgb, polar_a], dim=1))
        output_b = model(torch.cat([rgb, polar_b], dim=1))
    torch.testing.assert_close(output_a, output_b, rtol=0.0, atol=0.0)


def assert_inactive_gate_is_identity_and_not_called() -> None:
    torch.manual_seed(31)
    model = MultiscalePFEUNet(
        in_channels=4,
        out_channels=1,
        base_channels=8,
        enhancement="all",
        fusion="multiply",
        active_gate_levels=(2,),
    )
    call_count = 0

    def count_call(_module, _inputs, _output) -> None:
        nonlocal call_count
        call_count += 1

    handle = model.gate1.register_forward_hook(count_call)
    rgb_feature = torch.randn(2, 8, 16, 16, requires_grad=True)
    polar_feature = torch.randn(2, 8, 16, 16, requires_grad=True)
    output = model.enhance(1, model.gate1, rgb_feature, polar_feature)
    handle.remove()

    if output is not rgb_feature:
        raise AssertionError("Inactive level did not return the original RGB tensor object.")
    if call_count != 0:
        raise AssertionError(f"Inactive gate was called {call_count} time(s).")
    output.sum().backward()
    if any(parameter.grad is not None for parameter in model.gate1.parameters()):
        raise AssertionError("Inactive gate unexpectedly received gradients.")
    if polar_feature.grad is not None:
        raise AssertionError("Inactive level unexpectedly injected the polarization feature.")


def assert_active_gate_receives_gradient() -> None:
    torch.manual_seed(43)
    model = MultiscalePFEUNet(
        in_channels=4,
        out_channels=1,
        base_channels=8,
        enhancement="all",
        fusion="multiply",
        active_gate_levels=(2,),
    )
    rgb_feature = torch.randn(2, 16, 16, 16, requires_grad=True)
    polar_feature = torch.randn(2, 16, 16, 16, requires_grad=True)
    output = model.enhance(2, model.gate2, rgb_feature, polar_feature)
    output.square().mean().backward()
    if not any(parameter.grad is not None for parameter in model.gate2.parameters()):
        raise AssertionError("Active gate did not receive gradients.")
    if polar_feature.grad is None:
        raise AssertionError("Active gate did not propagate through the polarization feature.")


def assert_variant_configuration() -> None:
    for name, levels in CORE_VARIANTS.items():
        model = MultiscalePFEUNet(
            in_channels=4,
            out_channels=1,
            base_channels=8,
            enhancement="all",
            fusion="multiply",
            active_gate_levels=levels,
        )
        actual = tuple(sorted(model.active_gate_levels))
        if actual != levels:
            raise AssertionError(f"{name}: expected {levels}, found {actual}")


def main() -> None:
    assert_variant_configuration()
    assert_all_five_matches_reference()
    assert_no_gate_is_polarization_invariant()
    assert_inactive_gate_is_identity_and_not_called()
    assert_active_gate_receives_gradient()
    print("PASS: active_gate_levels implementation is identity-safe and M6 matches the reference.")


if __name__ == "__main__":
    main()
