from __future__ import annotations

import torch

from stress_common import (
    apply_perturbation,
    gaussian_blur_reflect,
    perturbation_seed,
    rotate_reflect,
    shift_reflect,
    validate_formal_sources,
)


def main() -> None:
    torch.manual_seed(11)
    images = torch.rand(2, 4, 32, 32)
    aux = images[:, 3:4]

    for direction in ("left", "right", "up", "down"):
        shifted = shift_reflect(aux, 4, direction)
        if shifted.shape != aux.shape or not torch.isfinite(shifted).all():
            raise AssertionError(f"Invalid reflected translation: {direction}")

    for angle in (-2.0, -0.5, 0.0, 0.5, 2.0):
        rotated = rotate_reflect(aux, angle)
        if rotated.shape != aux.shape or not torch.isfinite(rotated).all():
            raise AssertionError(f"Invalid reflected rotation: {angle}")

    blurred, kernel_size = gaussian_blur_reflect(aux, 0.5)
    if kernel_size != 5 or blurred.shape != aux.shape:
        raise AssertionError("Gaussian kernel selection is not 2*ceil(3*sigma)+1.")

    seed = perturbation_seed("noise", "0.02", 2)
    generator_a = torch.Generator(device="cpu").manual_seed(seed)
    generator_b = torch.Generator(device="cpu").manual_seed(seed)
    noisy_a, _ = apply_perturbation(images, "noise", 0.02, "", 2, generator_a)
    noisy_b, _ = apply_perturbation(images, "noise", 0.02, "", 2, generator_b)
    torch.testing.assert_close(noisy_a, noisy_b, rtol=0.0, atol=0.0)
    if noisy_a[:, 3:4].min() < 0 or noisy_a[:, 3:4].max() > 1:
        raise AssertionError("Noisy DoLP was not clipped to [0,1].")

    quantized, _ = apply_perturbation(images, "quantization", 2, "", 0, None)
    if torch.unique(quantized[:, 3:4]).numel() > 4:
        raise AssertionError("Two-bit DoLP has more than four quantization levels.")

    transformed, _ = apply_perturbation(images, "translation", 2, "left", 0, None)
    torch.testing.assert_close(transformed[:, :3], images[:, :3], rtol=0.0, atol=0.0)

    audit = validate_formal_sources()
    missing = [row for row in audit if row["status"] != "VERIFIED"]
    if missing:
        raise AssertionError(f"Formal checkpoint source audit failed: {missing}")
    print("PASS: stress transforms and all 15 formal checkpoint sources are verified.")


if __name__ == "__main__":
    main()
