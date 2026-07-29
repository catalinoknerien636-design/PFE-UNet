from __future__ import annotations

import argparse
import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[2]
REGISTRATION_CODE = EXTENSION_ROOT / "registration" / "code"
if str(REGISTRATION_CODE) not in sys.path:
    sys.path.insert(0, str(REGISTRATION_CODE))

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


OUTPUT_ROOT = EXTENSION_ROOT / "dolp_quality"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one five-seed DoLP quality stress unit.")
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
    for sigma in (0.0, 0.5, 1.0, 2.0, 4.0):
        rows.append(
            evaluate_condition(
                model,
                source,
                args.method,
                args.seed,
                device,
                loader,
                OUTPUT_ROOT,
                kind="blur",
                level=sigma,
            )
        )
    for std in (0.0, 0.01, 0.02, 0.05, 0.10):
        for realization in (1, 2, 3):
            rows.append(
                evaluate_condition(
                    model,
                    source,
                    args.method,
                    args.seed,
                    device,
                    loader,
                    OUTPUT_ROOT,
                    kind="noise",
                    level=std,
                    realization=realization,
                )
            )
    for bits in (8, 6, 4, 2):
        rows.append(
            evaluate_condition(
                model,
                source,
                args.method,
                args.seed,
                device,
                loader,
                OUTPUT_ROOT,
                kind="quantization",
                level=bits,
            )
        )
    write_csv(
        OUTPUT_ROOT / "runs" / f"{args.method}_seed{args.seed}" / "quality_suite_summary.csv",
        rows,
    )
    print(f"DoLP quality suite complete: {args.method} seed={args.seed}")


if __name__ == "__main__":
    main()
