from __future__ import annotations

import argparse
from pathlib import Path

from stress_common import (
    METHODS,
    SEEDS,
    evaluate_condition,
    load_model,
    make_loader,
    resolve_device,
    validate_formal_sources,
    write_csv,
)


EXTENSION_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = EXTENSION_ROOT / "registration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one five-seed registration stress unit.")
    parser.add_argument("--method", choices=list(METHODS), required=True)
    parser.add_argument("--seed", type=int, choices=list(SEEDS), required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "directml", "auto"], default="directml")
    parser.add_argument("--directml_device", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = validate_formal_sources()
    missing = [row for row in audit if row["status"] != "VERIFIED"]
    if missing:
        raise FileNotFoundError(f"Formal checkpoint audit failed: {missing}")
    write_csv(OUTPUT_ROOT / "formal_checkpoint_audit.csv", audit)

    device = resolve_device(args.device, args.directml_device)
    model, source = load_model(args.method, args.seed, device)
    loader = make_loader()
    rows = []
    rows.append(
        evaluate_condition(
            model,
            source,
            args.method,
            args.seed,
            device,
            loader,
            OUTPUT_ROOT,
            kind="clean",
            level=0,
        )
    )
    for pixels in (1, 2, 4, 8, 16):
        for direction in ("left", "right", "up", "down"):
            rows.append(
                evaluate_condition(
                    model,
                    source,
                    args.method,
                    args.seed,
                    device,
                    loader,
                    OUTPUT_ROOT,
                    kind="translation",
                    level=pixels,
                    direction=direction,
                )
            )
    for angle in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
        rows.append(
            evaluate_condition(
                model,
                source,
                args.method,
                args.seed,
                device,
                loader,
                OUTPUT_ROOT,
                kind="rotation",
                level=angle,
            )
        )
    write_csv(
        OUTPUT_ROOT / "runs" / f"{args.method}_seed{args.seed}" / "registration_suite_summary.csv",
        rows,
    )
    print(f"Registration suite complete: {args.method} seed={args.seed}")


if __name__ == "__main__":
    main()
