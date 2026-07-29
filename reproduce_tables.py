from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


DISPLAY = {
    "RGB+DoLP direct": "RGB+DoLP direct U-Net",
    "PFE-S0": "PFE-S0 multiply",
    "PFE-RGB-gray": "PFE-RGB-gray multiply",
    "PFE-1x1": "PFE 1 x 1 fusion control",
    "PFE-DoLP": "PFE-DoLP multiply",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/five_seed_run_level_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/Table_1_reproduced.csv"))
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keep = set(DISPLAY)
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] in keep:
            grouped[row["method"]].append(row)
    output = []
    for method in DISPLAY:
        method_rows = grouped[method]
        if len(method_rows) != 5:
            raise ValueError(f"Expected five rows for {method}, found {len(method_rows)}")
        item = {"Method": DISPLAY[method], "Seeds": 5}
        for label, source in [("IoU","test_iou"),("F1","test_f1"),("Precision","test_precision"),("Recall","test_recall"),("Image-averaged MAE","mae_image_mean")]:
            values = [float(row[source]) for row in method_rows]
            item[label] = f"{statistics.mean(values):.4f} +/- {statistics.stdev(values):.4f}"
        output.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    print(args.output)


if __name__ == "__main__":
    main()
