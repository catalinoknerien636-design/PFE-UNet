from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ.get("PFE_SOURCE_PROJECT_ROOT", REPOSITORY_ROOT)).resolve()
PYTHON = Path(sys.executable)
REG_ROOT = EXTENSION_ROOT / "registration"
QUALITY_ROOT = EXTENSION_ROOT / "dolp_quality"
MULTISCALE_LOG = EXTENSION_ROOT / "gate_placement" / "logs" / "queue.log"
SCRIPT_DIR = Path(__file__).resolve().parent
QUALITY_SCRIPT = QUALITY_ROOT / "code" / "run_quality_suite.py"
FIGURE_SCRIPT = EXTENSION_ROOT / "figures" / "plot_external_review_figures.py"
COMMAND_DIR = EXTENSION_ROOT / "commands"
LOG_DIR = REG_ROOT / "logs"
QUEUE_LOG = LOG_DIR / "queue.log"
METHODS = ("direct", "pfe_multiply", "pfe_1x1")
SEEDS = (42, 2026, 3407, 7819, 12011)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    with QUEUE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def wait_for_multiscale() -> None:
    log("WAITING FOR MULTISCALE QUEUE COMPLETE")
    while True:
        text = MULTISCALE_LOG.read_text(encoding="utf-8") if MULTISCALE_LOG.exists() else ""
        if "MULTISCALE QUEUE COMPLETE" in text:
            log("MULTISCALE COMPLETION DETECTED")
            return
        time.sleep(300)


def execute(name: str, command: list[str]) -> None:
    COMMAND_DIR.mkdir(parents=True, exist_ok=True)
    command_path = COMMAND_DIR / f"{name}.txt"
    command_path.write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
    run_log = LOG_DIR / f"{name}.log"
    log(f"START {name}")
    with run_log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{now()}] COMMAND {subprocess.list2cmdline(command)}\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=EXTENSION_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        log(f"ERROR {name} exit={result.returncode}; log={run_log}")
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    log(f"DONE {name}")


def registration_complete(method: str, seed: int) -> bool:
    return (
        row_count(
            REG_ROOT / "runs" / f"{method}_seed{seed}" / "registration_suite_summary.csv"
        )
        == 27
    )


def quality_complete(method: str, seed: int) -> bool:
    return (
        row_count(
            QUALITY_ROOT / "runs" / f"{method}_seed{seed}" / "quality_suite_summary.csv"
        )
        == 24
    )


def main() -> None:
    log("STRESS QUEUE START")
    wait_for_multiscale()
    for method in METHODS:
        for seed in SEEDS:
            name = f"registration_{method}_seed{seed}"
            if registration_complete(method, seed):
                log(f"SKIP COMPLETE {name}")
                continue
            execute(
                name,
                [
                    str(PYTHON),
                    str(SCRIPT_DIR / "run_registration_suite.py"),
                    "--method",
                    method,
                    "--seed",
                    str(seed),
                    "--device",
                    "directml",
                    "--directml_device",
                    "1",
                ],
            )
    for method in METHODS:
        for seed in SEEDS:
            name = f"quality_{method}_seed{seed}"
            if quality_complete(method, seed):
                log(f"SKIP COMPLETE {name}")
                continue
            execute(
                name,
                [
                    str(PYTHON),
                    str(QUALITY_SCRIPT),
                    "--method",
                    method,
                    "--seed",
                    str(seed),
                    "--device",
                    "directml",
                    "--directml_device",
                    "1",
                ],
            )
    execute(
        "stress_summarize",
        [str(PYTHON), str(SCRIPT_DIR / "summarize_stress.py")],
    )
    execute(
        "external_review_figures",
        [str(PYTHON), str(FIGURE_SCRIPT)],
    )
    log("STRESS QUEUE COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"STRESS QUEUE ABORTED: {type(exc).__name__}: {exc}")
        raise
