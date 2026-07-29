from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=Path("results/Table_1_reproduced.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/smoke_table1_iou.png"))
    args = parser.parse_args()
    with args.table.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["Method"].replace(" multiply", "") for row in rows]
    values = [float(row["IoU"].split("+/-")[0]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(len(values)), values, color=["#4C78A8", "#72B7B2", "#54A24B", "#F58518", "#2A9D8F"])
    ax.set_ylabel("IoU")
    ax.set_xticks(range(len(values)), names, rotation=20, ha="right")
    ax.set_ylim(0.38, 0.50)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    print(args.output)


if __name__ == "__main__":
    main()
