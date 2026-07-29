from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_ROOT = EXPERIMENT_ROOT.parents[0]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("PFE_SOURCE_PROJECT_ROOT", REPOSITORY_ROOT)).resolve()
EVIDENCE_ROOT = Path(os.environ.get("PFE_EVIDENCE_ROOT", REPOSITORY_ROOT)).resolve()
PYTHON = Path(sys.executable)
SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = EXPERIMENT_ROOT / "logs"
COMMAND_DIR = EXPERIMENT_ROOT / "commands"
QUEUE_LOG = LOG_DIR / "queue.log"
PLAN_PATH = EXPERIMENT_ROOT / "configs" / "experiment_plan.csv"

VARIANTS = {
    "M0": ("No active gate", ()),
    "M1": ("L1 only", (1,)),
    "M3": ("L5 only", (5,)),
    "M4": ("Shallow", (1, 2, 3)),
    "M5": ("Deep", (3, 4, 5)),
    "M6": ("All five levels", (1, 2, 3, 4, 5)),
}
PARTIAL_VARIANTS = ("M1", "M3", "M4", "M5")
PARTIAL_SEEDS = (42, 2026, 3407)
MAX_NEW_MULTISCALE_RUNS = 12

REUSED_M6 = {
    42: {
        "run": "grouped_pfe_unet_multiply_directml_50ep",
        "metrics_root": PROJECT_ROOT / "outputs" / "metrics",
        "checkpoint_root": PROJECT_ROOT / "checkpoints",
        "prediction_root": PROJECT_ROOT / "outputs" / "predictions",
    },
    2026: {
        "run": "grouped_pfe_unet_multiply_directml_50ep_seed2026",
        "metrics_root": PROJECT_ROOT / "outputs" / "metrics",
        "checkpoint_root": PROJECT_ROOT / "checkpoints",
        "prediction_root": PROJECT_ROOT / "outputs" / "predictions",
    },
    3407: {
        "run": "grouped_pfe_unet_multiply_directml_50ep_seed3407",
        "metrics_root": PROJECT_ROOT / "outputs" / "metrics",
        "checkpoint_root": PROJECT_ROOT / "checkpoints",
        "prediction_root": PROJECT_ROOT / "outputs" / "predictions",
    },
    7819: {
        "run": "ext_pfe_dolp_multiply_seed7819",
        "metrics_root": EVIDENCE_ROOT / "metrics",
        "checkpoint_root": EVIDENCE_ROOT / "checkpoints",
        "prediction_root": EVIDENCE_ROOT / "predictions",
    },
    12011: {
        "run": "ext_pfe_dolp_multiply_seed12011",
        "metrics_root": EVIDENCE_ROOT / "metrics",
        "checkpoint_root": EVIDENCE_ROOT / "checkpoints",
        "prediction_root": EVIDENCE_ROOT / "predictions",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{utc_now()}] {message}"
    with QUEUE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def levels_cli(levels: tuple[int, ...]) -> str:
    return ",".join(f"L{level}" for level in levels)


def run_name(variant: str, seed: int) -> str:
    suffix = levels_cli(VARIANTS[variant][1]).replace(",", "_") or "none"
    return f"multiscale_{variant}_{suffix}_seed{seed}"


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def prediction_count(path: Path) -> int:
    return len(list(path.glob("*.png"))) if path.exists() else 0


def formal_complete(name: str) -> bool:
    metrics = EXPERIMENT_ROOT / "metrics" / name
    return all(
        [
            count_rows(metrics / "history.csv") == 50,
            (metrics / "config.json").exists(),
            (metrics / "test_metrics.csv").exists(),
            count_rows(metrics / "test_per_sample.csv") == 230,
            (metrics / "cod_metrics.csv").exists(),
            count_rows(metrics / "cod_per_sample.csv") == 230,
            (EXPERIMENT_ROOT / "checkpoints" / name / "best.pt").exists(),
            prediction_count(EXPERIMENT_ROOT / "predictions" / name) == 230,
        ]
    )


def write_command(name: str, stage: str, command: list[str]) -> None:
    COMMAND_DIR.mkdir(parents=True, exist_ok=True)
    path = COMMAND_DIR / f"{name}_{stage}.txt"
    path.write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")


def execute(name: str, stage: str, command: list[str]) -> None:
    log_path = LOG_DIR / f"{name}_{stage}.log"
    write_command(name, stage, command)
    log(f"START {name} {stage}")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{utc_now()}] COMMAND {subprocess.list2cmdline(command)}\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=EXPERIMENT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        log(f"ERROR {name} {stage} exit={result.returncode}; log={log_path}")
        raise RuntimeError(f"{name} {stage} failed with exit code {result.returncode}")
    log(f"DONE {name} {stage}")


def train_and_evaluate(variant: str, seed: int) -> None:
    if variant not in PARTIAL_VARIANTS or seed not in PARTIAL_SEEDS:
        raise RuntimeError(
            "Training outside the author-approved 12-run scope is forbidden: "
            f"variant={variant}, seed={seed}"
        )
    name = run_name(variant, seed)
    levels = VARIANTS[variant][1]
    if formal_complete(name):
        log(f"SKIP COMPLETE {name}")
        return

    metrics = EXPERIMENT_ROOT / "metrics" / name
    checkpoint_dir = EXPERIMENT_ROOT / "checkpoints" / name
    history = metrics / "history.csv"
    latest = checkpoint_dir / "latest.pt"
    best = checkpoint_dir / "best.pt"
    history_rows = count_rows(history)

    if history_rows < 50:
        train_command = [
            str(PYTHON),
            str(SCRIPT_DIR / "train_multiscale.py"),
            "--model",
            "pfe_unet",
            "--input_mode",
            "rgb_dop",
            "--pfe_fusion",
            "multiply",
            "--active_gate_levels",
            levels_cli(levels),
            "--run_name",
            name,
            "--extension_root",
            str(EXPERIMENT_ROOT),
            "--epochs",
            "50",
            "--batch_size",
            "4",
            "--base_channels",
            "32",
            "--seed",
            str(seed),
            "--device",
            "directml",
            "--directml_device",
            "1",
        ]
        if history_rows > 0:
            if not latest.exists():
                raise FileNotFoundError(f"History exists without resumable checkpoint: {history}")
            train_command.extend(["--resume_checkpoint", str(latest)])
        execute(name, "train", train_command)

    if not best.exists():
        raise FileNotFoundError(f"Missing validation-selected checkpoint: {best}")

    if count_rows(metrics / "test_per_sample.csv") != 230 or prediction_count(
        EXPERIMENT_ROOT / "predictions" / name
    ) != 230:
        evaluate_command = [
            str(PYTHON),
            str(SCRIPT_DIR / "evaluate_multiscale.py"),
            "--checkpoint",
            str(best),
            "--run_name",
            name,
            "--model",
            "pfe_unet",
            "--input_mode",
            "rgb_dop",
            "--pfe_fusion",
            "multiply",
            "--active_gate_levels",
            levels_cli(levels),
            "--extension_root",
            str(EXPERIMENT_ROOT),
            "--batch_size",
            "4",
            "--base_channels",
            "32",
            "--device",
            "directml",
            "--directml_device",
            "1",
        ]
        execute(name, "evaluate", evaluate_command)

    if count_rows(metrics / "cod_per_sample.csv") != 230:
        cod_command = [
            str(PYTHON),
            str(SCRIPT_DIR / "cod_metrics_multiscale.py"),
            "--pred_dir",
            str(EXPERIMENT_ROOT / "predictions" / name),
            "--run_name",
            name,
            "--out_dir",
            str(metrics),
        ]
        execute(name, "cod_metrics", cod_command)

    if not formal_complete(name):
        raise RuntimeError(f"Artifact verification failed after run: {name}")
    log(f"VERIFIED {name}")


def prepare_reused_m0_seed42() -> None:
    name = run_name("M0", 42)
    metrics = EXPERIMENT_ROOT / "metrics" / name
    predictions = EXPERIMENT_ROOT / "predictions" / name
    source_run = "grouped_pfe_unet_none_directml_50ep"
    source_metrics = PROJECT_ROOT / "outputs" / "metrics" / source_run
    source_checkpoint = PROJECT_ROOT / "checkpoints" / source_run / "best.pt"
    metrics.mkdir(parents=True, exist_ok=True)

    if not (metrics / "history.csv").exists():
        shutil.copy2(source_metrics / "history.csv", metrics / "history.csv")
    if not (metrics / "config.json").exists():
        source_config = json.loads((source_metrics / "config.json").read_text(encoding="utf-8"))
        source_config.update(
            {
                "reuse_source_run": source_run,
                "reuse_source_checkpoint": str(source_checkpoint),
                "identity_bypass_verified_by": str(
                    EXPERIMENT_ROOT / "code" / "tests" / "validate_active_gate_levels.py"
                ),
                "active_gate_levels_normalized": [],
                "notes": "Historical enhancement=none is equivalent to active_gate_levels=[]; fusion is never called.",
            }
        )
        (metrics / "config.json").write_text(
            json.dumps(source_config, indent=2), encoding="utf-8"
        )

    if count_rows(metrics / "test_per_sample.csv") != 230 or prediction_count(predictions) != 230:
        execute(
            name,
            "evaluate_reused",
            [
                str(PYTHON),
                str(SCRIPT_DIR / "evaluate_multiscale.py"),
                "--checkpoint",
                str(source_checkpoint),
                "--run_name",
                name,
                "--model",
                "pfe_unet",
                "--input_mode",
                "rgb_dop",
                "--pfe_fusion",
                "multiply",
                "--active_gate_levels",
                "",
                "--extension_root",
                str(EXPERIMENT_ROOT),
                "--batch_size",
                "4",
                "--base_channels",
                "32",
                "--device",
                "directml",
                "--directml_device",
                "1",
            ],
        )
    if count_rows(metrics / "cod_per_sample.csv") != 230:
        execute(
            name,
            "cod_metrics_reused",
            [
                str(PYTHON),
                str(SCRIPT_DIR / "cod_metrics_multiscale.py"),
                "--pred_dir",
                str(predictions),
                "--run_name",
                name,
                "--out_dir",
                str(metrics),
            ],
        )
    (metrics / "reused_artifact.json").write_text(
        json.dumps(
            {
                "source_run": source_run,
                "source_checkpoint": str(source_checkpoint),
                "source_history": str(source_metrics / "history.csv"),
                "verified_utc": utc_now(),
                "checkpoint_not_duplicated": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"VERIFIED REUSED {name}")


def verify_reused_m6() -> None:
    for seed, source in REUSED_M6.items():
        name = source["run"]
        metrics = source["metrics_root"] / name
        checkpoint = source["checkpoint_root"] / name / "best.pt"
        predictions = source["prediction_root"] / name
        missing = []
        if count_rows(metrics / "history.csv") != 50:
            missing.append("history")
        if not (metrics / "test_metrics.csv").exists():
            missing.append("test_metrics")
        if not checkpoint.exists():
            missing.append("best_checkpoint")
        if prediction_count(predictions) != 230:
            missing.append("predictions")
        if missing:
            raise RuntimeError(f"Reused M6 seed {seed} missing: {missing}")
        log(f"VERIFIED REUSED M6 seed={seed} run={name}")


def best_validation_iou(name: str) -> float:
    path = EXPERIMENT_ROOT / "metrics" / name / "history.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(float(row["val_iou"]) for row in csv.DictReader(handle))


def select_best_partial() -> str:
    rows = []
    for variant in PARTIAL_VARIANTS:
        values = [best_validation_iou(run_name(variant, seed)) for seed in PARTIAL_SEEDS]
        rows.append(
            {
                "variant": variant,
                "active_gate_levels": levels_cli(VARIANTS[variant][1]),
                "seed_best_validation_iou": dict(zip(PARTIAL_SEEDS, values)),
                "mean_best_validation_iou": sum(values) / len(values),
            }
        )
    rows.sort(key=lambda row: (-row["mean_best_validation_iou"], row["variant"]))
    decision = {
        "selection_metric": "mean of per-seed best validation IoU",
        "test_metrics_read_for_selection": False,
        "tie_break_order": list(PARTIAL_VARIANTS),
        "selected_variant": rows[0]["variant"],
        "selected_active_gate_levels": rows[0]["active_gate_levels"],
        "candidates": rows,
        "selected_utc": utc_now(),
    }
    path = EXPERIMENT_ROOT / "configs" / "partial_variant_selection.json"
    path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    log(
        "SELECTED "
        f"{decision['selected_variant']} by validation-only mean "
        f"{rows[0]['mean_best_validation_iou']:.8f}"
    )
    return str(decision["selected_variant"])


def snapshot_plan_status(selected_partial: str | None = None) -> None:
    with PLAN_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        variant = row["variant"]
        seed = int(row["seed"])
        if variant == "M6":
            row["status"] = "VERIFIED_REUSED"
        elif variant == "M0":
            if seed == 42:
                name = run_name(variant, seed)
                row["status"] = (
                    "VERIFIED_REUSED"
                    if count_rows(EXPERIMENT_ROOT / "metrics" / name / "cod_per_sample.csv")
                    == 230
                    else "PENDING_REUSE_VERIFICATION"
                )
            else:
                row["status"] = "REUSE_SOURCE_REQUIRED"
        elif variant in PARTIAL_VARIANTS:
            name = run_name(variant, seed)
            if formal_complete(name):
                row["status"] = "VERIFIED"
            elif count_rows(EXPERIMENT_ROOT / "metrics" / name / "history.csv") > 0:
                row["status"] = "IN_PROGRESS"
    with PLAN_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    approved_runs = [
        (variant, seed)
        for variant in PARTIAL_VARIANTS
        for seed in PARTIAL_SEEDS
    ]
    if len(approved_runs) != MAX_NEW_MULTISCALE_RUNS:
        raise RuntimeError(
            f"Approved multiscale scope must contain exactly "
            f"{MAX_NEW_MULTISCALE_RUNS} runs, found {len(approved_runs)}"
        )
    log(
        "MULTISCALE QUEUE START "
        f"(author-corrected scope: {len(approved_runs)} new runs; "
        "M0 and M6 reuse only)"
    )
    prepare_reused_m0_seed42()
    verify_reused_m6()
    snapshot_plan_status()

    for variant in PARTIAL_VARIANTS:
        for seed in PARTIAL_SEEDS:
            train_and_evaluate(variant, seed)
            snapshot_plan_status()

    selected = select_best_partial()
    snapshot_plan_status(selected)

    execute(
        "multiscale_complexity",
        "measure",
        [
            str(PYTHON),
            str(SCRIPT_DIR / "complexity_multiscale.py"),
            "--extension_root",
            str(EXPERIMENT_ROOT),
            "--device",
            "directml",
            "--directml_device",
            "1",
        ],
    )
    execute(
        "multiscale_summary",
        "summarize",
        [str(PYTHON), str(SCRIPT_DIR / "summarize_multiscale.py")],
    )
    snapshot_plan_status(selected)
    log(
        "MULTISCALE QUEUE COMPLETE "
        f"({MAX_NEW_MULTISCALE_RUNS} author-approved new runs; no expansion)"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"QUEUE ABORTED: {type(exc).__name__}: {exc}")
        raise
